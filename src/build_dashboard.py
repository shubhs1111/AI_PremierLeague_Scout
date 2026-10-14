"""Inject the pipeline output into the dashboard template.

    python -m src.build_dashboard   ->  outputs/dashboard.html
"""
from .config import OUTPUTS, ROOT


def build() -> str:
    template = (ROOT / "dashboard" / "template.html").read_text()
    data = (OUTPUTS / "dashboard_data.json").read_text().replace("</", "<\\/")
    html = template.replace("/*__DATA__*/null", data)
    out = OUTPUTS / "dashboard.html"
    out.write_text(html)
    return str(out)


if __name__ == "__main__":
    print(build())
