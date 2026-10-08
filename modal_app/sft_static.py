"""
Static filter shared by the SFT ingest scripts (results/v3/sft/README.md, sft_plan.md Phase 1 step 2).
Local CPU only, no torch/triton import needed.

    static_check(code, fmt="modelnew"|"autokernel", ref_names=("Model",)) -> dict

Verdict fields:
  parses           ast.parse succeeds
  jit_kernels      number of functions decorated with triton.jit / jit (directly or under autotune/heuristics)
  launched         at least one jit kernel is launched as  name[grid](...)  (launched_kernels lists them)
  entry_found      ModelNew.forward (modelnew) / kernel_fn (autokernel) exists
  forbidden        reasons the row is rejected (empty list = clean):
                     extern_kernels      any `extern_kernels` reference (cuBLAS/cuDNN fallback)
                     try                 any try: block (fallback pattern)
                     inherits_ref        ModelNew inherits from the reference class
                     @                   matmul operator on the forward path
                     <dotted call>       torch.matmul/mm/bmm/addmm/baddbmm/einsum, torch.softmax,
                                         torch.layer_norm, F.* / torch.nn.functional.* / nn.functional.*,
                                         torch.ops.aten.* (eager fallback) on the forward path
                     self.training       train/eval branch on the forward path
                     no_jit / not_launched / no_entry / syntax_error
  tl_symbols       sorted dotted names rooted at a triton.language alias (tl.load, tl.math.exp, ...)
  ok               parses and jit_kernels>=1 and launched and entry_found and not forbidden

The forward path = the entry function plus every module-level function / same-class method it
calls, transitively (Inductor's forward -> call(args) -> kernel launches).
"""
import ast
import hashlib

FORBIDDEN_CALLS = {
    "torch.matmul", "torch.mm", "torch.bmm", "torch.addmm", "torch.baddbmm", "torch.einsum",
    "torch.softmax", "torch.log_softmax", "torch.layer_norm", "torch.nn.functional.linear",
}
FORBIDDEN_PREFIXES = ("torch.nn.functional.", "F.", "nn.functional.", "torch.ops.aten.")


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def ast_key(code):
    """Normalised AST (comments/docstrings/formatting stripped); same as val_eval.ast_key."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()


def _is_jit_decorator(d):
    if isinstance(d, ast.Call):
        d = d.func
    name = dotted(d) or ""
    return name in ("triton.jit", "jit", "triton.runtime.jit")


def _tl_aliases(tree):
    al = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "triton.language":
                    al.add(a.asname or "triton.language")
        elif isinstance(n, ast.ImportFrom):
            if n.module == "triton":
                for a in n.names:
                    if a.name == "language":
                        al.add(a.asname or "language")
    return al or {"tl"}


def static_check(code, fmt="modelnew", ref_names=("Model",)):
    out = {"parses": False, "jit_kernels": 0, "launched": False, "launched_kernels": [],
           "entry_found": False, "forbidden": [], "tl_symbols": [], "ok": False}
    if "extern_kernels" in code:
        out["forbidden"].append("extern_kernels")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        out["forbidden"].append("syntax_error")
        return out
    out["parses"] = True

    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    jit = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(_is_jit_decorator(d) for d in n.decorator_list):
            jit.add(n.name)
    out["jit_kernels"] = len(jit)

    launched = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript):
            v = n.func.value
            if isinstance(v, ast.Name) and v.id in jit:
                launched.add(v.id)
    out["launched_kernels"] = sorted(launched)
    out["launched"] = bool(launched)

    try_types = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, "TryStar") else ())
    if any(isinstance(n, try_types) for n in ast.walk(tree)):
        out["forbidden"].append("try")

    # entry points
    roots, cls_methods = [], {}
    if fmt == "modelnew":
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name == "ModelNew":
                cls_methods = {m.name: m for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                if "forward" in cls_methods:
                    roots.append(cls_methods["forward"])
                for b in n.bases:  # the reference class itself (a bare name), not e.g. nn.LayerNorm
                    if isinstance(b, ast.Name) and b.id in ref_names:
                        out["forbidden"].append("inherits_ref")
    else:
        if "kernel_fn" in funcs:
            roots.append(funcs["kernel_fn"])
    out["entry_found"] = bool(roots)

    # forward-path closure
    seen, stack, path = set(), list(roots), []
    while stack:
        f = stack.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        path.append(f)
        for n in ast.walk(f):
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Name) and fn.id in funcs and fn.id not in jit:
                    stack.append(funcs[fn.id])
                elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self" \
                        and fn.attr in cls_methods:
                    stack.append(cls_methods[fn.attr])
    bad = []
    for f in path:
        for n in ast.walk(f):
            if isinstance(n, (ast.BinOp, ast.AugAssign)) and isinstance(n.op, ast.MatMult):
                bad.append("@")
            elif isinstance(n, ast.Call):
                name = dotted(n.func)
                if name and (name in FORBIDDEN_CALLS or name.startswith(FORBIDDEN_PREFIXES)):
                    bad.append(name)
            elif isinstance(n, ast.Attribute) and n.attr == "training" and isinstance(n.value, ast.Name) \
                    and n.value.id == "self":
                bad.append("self.training")
    for b in bad:
        if b not in out["forbidden"]:
            out["forbidden"].append(b)

    tla = _tl_aliases(tree)
    syms = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            d = dotted(n)
            if d and d.split(".")[0] in tla and "." in d:
                syms.add("tl." + d.split(".", 1)[1])
    # keep only maximal chains (drop tl.math when tl.math.exp is present)
    out["tl_symbols"] = sorted(s for s in syms if not any(o != s and o.startswith(s + ".") for o in syms))

    if not jit:
        out["forbidden"].append("no_jit")
    elif not launched:
        out["forbidden"].append("not_launched")
    if not roots:
        out["forbidden"].append("no_entry")
    out["ok"] = out["parses"] and out["jit_kernels"] >= 1 and out["launched"] and out["entry_found"] \
        and not out["forbidden"]
    return out
