"""Canh bất biến: mọi thứ app nạp lúc khởi động đều được bake vào image.

Đây là lỗi đã xảy ra hai lần và cả hai lần đều chỉ lộ ra khi container khởi
động trên máy chủ, không lộ lúc build cũng không lộ lúc chạy test - vì máy dev
có sẵn cache Hugging Face nên `from_pretrained` vẫn tải được, còn trên máy chủ
`HF_HUB_OFFLINE=1` biến file thiếu thành một TypeError không nêu tên model nào.

Hai test dưới đây kiểm bằng cách đọc mã nguồn chứ không chạy: chúng phải chạy
được ở CI nơi không có cache lẫn mạng.
"""

import ast
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent


def _cac_lenh_nap(cay: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(cay)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_pretrained"
    ]


def test_chi_load_models_duoc_goi_from_pretrained():
    """Một lời gọi from_pretrained nằm ngoài load_models là một model sẽ không
    được bake, và container sẽ chết lúc khởi động."""
    cay = ast.parse((GOC / "app" / "main.py").read_text(encoding="utf-8"))
    ham = next(
        n
        for n in cay.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "load_models"
    )
    trong_ham = {id(c) for c in _cac_lenh_nap(ham)}
    ngoai = [c for c in _cac_lenh_nap(cay) if id(c) not in trong_ham]
    assert not ngoai, (
        "from_pretrained được gọi ngoài load_models ở dòng "
        + ", ".join(str(c.lineno) for c in ngoai)
        + " - thêm nó vào load_models, nếu không nó sẽ không được bake."
    )


def test_dockerfile_bake_bang_chinh_load_models():
    """Dockerfile không được liệt kê lại danh sách model bằng tay."""
    df = (GOC / "Dockerfile").read_text(encoding="utf-8")
    lenh = [d for d in df.splitlines() if d.startswith("RUN") and "python -c" in d]
    assert lenh, "không thấy bước bake nào trong Dockerfile"
    assert any("load_models" in d for d in lenh), (
        "bước bake phải gọi app.main.load_models"
    )
    assert not any("from_pretrained" in d for d in lenh), (
        "Dockerfile đang liệt kê lại model bằng tay - đó chính là chỗ đã lệch hai lần"
    )


def test_load_models_tra_ve_du_khoa_ma_lifespan_gan():
    """Tên khoá trả về phải khớp với thuộc tính app.state mà các route đọc."""
    src = (GOC / "app" / "main.py").read_text(encoding="utf-8")
    cay = ast.parse(src)
    ham = next(
        n
        for n in cay.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "load_models"
    )
    tra_ve = next(n for n in ast.walk(ham) if isinstance(n, ast.Return))
    khoa = {k.value for k in tra_ve.value.keys if isinstance(k, ast.Constant)}

    doc = {
        n.attr
        for n in ast.walk(cay)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Attribute)
        and n.value.attr == "state"
        and isinstance(n.ctx, ast.Load)
    }
    thieu = doc - khoa
    assert not thieu, f"route đọc app.state.{thieu} nhưng load_models không tạo ra"
