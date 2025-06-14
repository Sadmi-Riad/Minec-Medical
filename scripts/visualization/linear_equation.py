from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView

# HTML template with MathJax to render LaTeX equations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
      body {
        font-size: 18px;
        text-align: center;
        margin: 20px;
      }
    </style>
  </head>
  <body>
    $$ {equation} $$
  </body>
</html>
"""

class EquationDialogEsti(QDialog):
    def __init__(self, latex_equation, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linear Regression Equation")
        self.resize(600, 400)
        
        html_content = HTML_TEMPLATE.replace("{equation}", latex_equation)
        
        self.view = QWebEngineView()
        self.view.setHtml(html_content)
        
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

def linear_regression_equation(model, feature_names, terms_per_line=4):
    if not hasattr(model, "intercept_") or not hasattr(model, "coef_"):
        return "Model not trained or incompatible."
    
    intercept = model.intercept_
    coefs = model.coef_
    
    eq = r"\hat{y} = "
    eq += f"{intercept:.4f}"
    
    for i, coef in enumerate(coefs):
        if i % terms_per_line == 0 and i != 0:
            eq += r"\\ "
        sign = "+" if coef >= 0 else "-"
        val = abs(coef)
        eq += f" {sign} {val:.4f}\\cdot {feature_names[i]}"
    
    return eq