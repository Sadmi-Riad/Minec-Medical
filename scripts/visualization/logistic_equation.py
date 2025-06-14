from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtWebEngineWidgets import QWebEngineView

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

class EquationDialog(QDialog):
    def __init__(self, latex_equation, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Logistic Regression Equation")
        self.resize(600, 400)
        
        html_content = HTML_TEMPLATE.replace("{equation}", latex_equation)
        
        self.view = QWebEngineView()
        self.view.setHtml(html_content)
        
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)
        
        
def logistic_regression_equation(model, feature_names, terms_per_line=4):
    if not hasattr(model, "intercept_") or not hasattr(model, "coef_"):
        return "model not trained"
    
    w0 = model.intercept_[0]
    coefs = model.coef_[0]
    
    # On commence la formule
    eq = f"y = {w0:.4f}"
    
    for i, w in enumerate(coefs):
        # saut de ligne tous les terms_per_line termes (optionnel)
        if i and i % terms_per_line == 0:
            eq += r" \\"
        sign = "+" if w >= 0 else "-"
        eq += f" {sign} {abs(w):.4f}\\,\\times\\,{feature_names[i]}"
    
    # Encapsulation LaTeX
    return  eq 
