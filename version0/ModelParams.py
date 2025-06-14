from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDoubleSpinBox, QComboBox, QPushButton, QSpinBox, QLabel, QSpacerItem, QSizePolicy, QHBoxLayout, QCheckBox)
from PyQt5.QtCore import Qt

class DynamicParameterDialog(QDialog):
    def __init__(self, parent=None, model_type="decision_tree", saved_params=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Model Parameters")
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        # Title label
        title_label = QLabel(f"Parameters for {model_type.replace('_', ' ').title()}")
        title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #35586D;
            margin-bottom: 20px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)

        # Form layout
        self.form_layout = QFormLayout()
        self.form_layout.setVerticalSpacing(12)
        self.form_layout.setHorizontalSpacing(25)
        self.form_layout.setLabelAlignment(Qt.AlignLeft)

        # Model type and widget initialization
        self.model_type = model_type
        self.param_widgets = {}
        self.saved_params = saved_params if saved_params else {}

        # Dynamically load parameters
        self.load_params()

        # Add form layout to main layout
        self.main_layout.addLayout(self.form_layout)

        # Spacer to push buttons down
        self.main_layout.addSpacerItem(QSpacerItem(10, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Button layout
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)
        self.button_layout.setContentsMargins(0, 0, 0, 0)

        # Create buttons
        self.save_button = QPushButton("Save Parameters")
        self.cancel_button = QPushButton("Cancel")
        
        # Button style
        button_style = """
        QPushButton {
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 14px;
            border: none;
            min-width: 120px;
        }
        QPushButton:hover {
            opacity: 0.9;
        }
        """
        self.save_button.setStyleSheet(button_style + """
        QPushButton {
            background-color: #35586D;
            color: white;
        }
        """)
        self.cancel_button.setStyleSheet(button_style + """
        QPushButton {
            background-color: #EE6843;
            color: white;
        }
        """)

        # Add buttons to button layout
        self.button_layout.addWidget(self.cancel_button)
        self.button_layout.addWidget(self.save_button)

        # Add button layout to main layout
        self.main_layout.addLayout(self.button_layout)

        # Connect signals
        self.save_button.clicked.connect(self.save_params)
        self.cancel_button.clicked.connect(self.reject)

    def load_params(self):
        """Loads model parameters dynamically based on selected model type."""
        widget_style = """
        QWidget {
            font-size: 13px;
        }
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
            padding: 5px;
            min-width: 150px;
        }
        """
        
        # Parameters for Decision Tree
        if self.model_type == "decision_tree":
            self.add_decision_tree_params()

        # Parameters for Linear Regression
        elif self.model_type == "linear_regression":
            self.add_linear_regression_params()

        # Parameters for Logistic Regression
        elif self.model_type == "logistic_regression":
            self.add_logistic_regression_params()

        # Parameters for KMeans
        elif self.model_type == "k_means":
            self.add_kmeans_params()

        # Parameters for EM Algorithm (Gaussian Mixture)
        elif self.model_type == "em_algorithm":
            self.add_em_algorithm_params()

    def add_decision_tree_params(self):
        """Adds parameters for Decision Tree classifier/regressor."""
        # Criterion
        self.param_widgets['criterion'] = QComboBox()
        self.param_widgets['criterion'].addItems(['gini', 'entropy', 'log_loss'])
        self.param_widgets['criterion'].setCurrentText(self.saved_params.get('criterion', 'gini'))
        self.form_layout.addRow("Criterion:", self.param_widgets['criterion'])

        # Splitter
        self.param_widgets['splitter'] = QComboBox()
        self.param_widgets['splitter'].addItems(['best', 'random'])
        self.param_widgets['splitter'].setCurrentText(self.saved_params.get('splitter', 'best'))
        self.form_layout.addRow("Splitter:", self.param_widgets['splitter'])

        # Max Depth
        self.param_widgets['max_depth'] = QSpinBox()
        self.param_widgets['max_depth'].setRange(1, 100)
        self.param_widgets['max_depth'].setValue(self.saved_params.get('max_depth', None) or 0)
        self.param_widgets['max_depth'].setSpecialValueText("None")
        self.form_layout.addRow("Max Depth:", self.param_widgets['max_depth'])

        # Min Samples Split
        self.param_widgets['min_samples_split'] = QSpinBox()
        self.param_widgets['min_samples_split'].setRange(2, 20)
        self.param_widgets['min_samples_split'].setValue(self.saved_params.get('min_samples_split', 2))
        self.form_layout.addRow("Min Samples Split:", self.param_widgets['min_samples_split'])

        # Min Samples Leaf
        self.param_widgets['min_samples_leaf'] = QSpinBox()
        self.param_widgets['min_samples_leaf'].setRange(1, 20)
        self.param_widgets['min_samples_leaf'].setValue(self.saved_params.get('min_samples_leaf', 1))
        self.form_layout.addRow("Min Samples Leaf:", self.param_widgets['min_samples_leaf'])

        # Min Weight Fraction Leaf
        self.param_widgets['min_weight_fraction_leaf'] = QDoubleSpinBox()
        self.param_widgets['min_weight_fraction_leaf'].setRange(0.0, 0.5)
        self.param_widgets['min_weight_fraction_leaf'].setSingleStep(0.01)
        self.param_widgets['min_weight_fraction_leaf'].setValue(self.saved_params.get('min_weight_fraction_leaf', 0.0))
        self.form_layout.addRow("Min Weight Fraction Leaf:", self.param_widgets['min_weight_fraction_leaf'])

        # Max Features
        self.param_widgets['max_features'] = QComboBox()
        self.param_widgets['max_features'].addItems(['None', 'sqrt', 'log2'])
        max_features = self.saved_params.get('max_features', None)
        self.param_widgets['max_features'].setCurrentText(str(max_features) if max_features else 'None')
        self.form_layout.addRow("Max Features:", self.param_widgets['max_features'])

        # Max Leaf Nodes
        self.param_widgets['max_leaf_nodes'] = QSpinBox()
        self.param_widgets['max_leaf_nodes'].setRange(2, 1000)
        self.param_widgets['max_leaf_nodes'].setValue(self.saved_params.get('max_leaf_nodes', None) or 0)
        self.param_widgets['max_leaf_nodes'].setSpecialValueText("None")
        self.form_layout.addRow("Max Leaf Nodes:", self.param_widgets['max_leaf_nodes'])

        # CCP Alpha
        self.param_widgets['ccp_alpha'] = QDoubleSpinBox()
        self.param_widgets['ccp_alpha'].setRange(0.0, 1.0)
        self.param_widgets['ccp_alpha'].setSingleStep(0.01)
        self.param_widgets['ccp_alpha'].setValue(self.saved_params.get('ccp_alpha', 0.0))
        self.form_layout.addRow("CCP Alpha:", self.param_widgets['ccp_alpha'])

    def add_linear_regression_params(self):
        """Adds parameters for Linear Regression."""
        # Fit Intercept
        self.param_widgets['fit_intercept'] = QCheckBox()
        self.param_widgets['fit_intercept'].setChecked(self.saved_params.get('fit_intercept', True))
        self.form_layout.addRow("Fit Intercept:", self.param_widgets['fit_intercept'])

        # Positive
        self.param_widgets['positive'] = QCheckBox()
        self.param_widgets['positive'].setChecked(self.saved_params.get('positive', False))
        self.form_layout.addRow("Positive Coefficients:", self.param_widgets['positive'])

    def add_logistic_regression_params(self):
        """Adds parameters for Logistic Regression."""
        # Penalty
        self.param_widgets['penalty'] = QComboBox()
        self.param_widgets['penalty'].addItems(['l2','None'])
        self.param_widgets['penalty'].setCurrentText(self.saved_params.get('penalty', 'l2'))
        self.form_layout.addRow("Penalty:", self.param_widgets['penalty'])

        # C (Inverse Regularization)
        self.param_widgets['C'] = QDoubleSpinBox()
        self.param_widgets['C'].setRange(0.001, 1000)
        self.param_widgets['C'].setSingleStep(0.1)
        self.param_widgets['C'].setValue(self.saved_params.get('C', 1.0))
        self.form_layout.addRow("Inverse Regularization (C):", self.param_widgets['C'])

        # Solver
        self.param_widgets['solver'] = QComboBox()
        self.param_widgets['solver'].addItems(['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'])
        self.param_widgets['solver'].setCurrentText(self.saved_params.get('solver', 'lbfgs'))
        self.form_layout.addRow("Solver:", self.param_widgets['solver'])

        # Max Iterations
        self.param_widgets['max_iter'] = QSpinBox()
        self.param_widgets['max_iter'].setRange(10, 10000)
        self.param_widgets['max_iter'].setValue(self.saved_params.get('max_iter', 100))
        self.form_layout.addRow("Max Iterations:", self.param_widgets['max_iter'])

        # Fit Intercept
        self.param_widgets['fit_intercept'] = QCheckBox()
        self.param_widgets['fit_intercept'].setChecked(self.saved_params.get('fit_intercept', True))
        self.form_layout.addRow("Fit Intercept:", self.param_widgets['fit_intercept'])

    def add_kmeans_params(self):
        """Adds parameters for KMeans clustering."""

        # Initialization Method
        self.param_widgets['init'] = QComboBox()
        self.param_widgets['init'].addItems(['k-means++', 'random'])
        self.param_widgets['init'].setCurrentText(self.saved_params.get('init', 'k-means++'))
        self.form_layout.addRow("Initialization Method:", self.param_widgets['init'])

        # Max Iterations
        self.param_widgets['max_iter'] = QSpinBox()
        self.param_widgets['max_iter'].setRange(10, 1000)
        self.param_widgets['max_iter'].setValue(self.saved_params.get('max_iter', 300))
        self.form_layout.addRow("Max Iterations:", self.param_widgets['max_iter'])

        # Algorithm
        self.param_widgets['algorithm'] = QComboBox()
        self.param_widgets['algorithm'].addItems(['auto', 'full', 'elkan'])
        self.param_widgets['algorithm'].setCurrentText(self.saved_params.get('algorithm', 'auto'))
        self.form_layout.addRow("Algorithm:", self.param_widgets['algorithm'])

    def add_em_algorithm_params(self):
        """Adds parameters for EM Algorithm (Gaussian Mixture)."""
        # Number of Components
        self.param_widgets['n_components'] = QSpinBox()
        self.param_widgets['n_components'].setRange(1, 100)
        self.param_widgets['n_components'].setValue(self.saved_params.get('n_components', 1))
        self.form_layout.addRow("Number of Components:", self.param_widgets['n_components'])

        # Covariance Type
        self.param_widgets['covariance_type'] = QComboBox()
        self.param_widgets['covariance_type'].addItems(['full', 'tied', 'diag', 'spherical'])
        self.param_widgets['covariance_type'].setCurrentText(self.saved_params.get('covariance_type', 'full'))
        self.form_layout.addRow("Covariance Type:", self.param_widgets['covariance_type'])

        # Max Iterations
        self.param_widgets['max_iter'] = QSpinBox()
        self.param_widgets['max_iter'].setRange(10, 1000)
        self.param_widgets['max_iter'].setValue(self.saved_params.get('max_iter', 100))
        self.form_layout.addRow("Max Iterations:", self.param_widgets['max_iter'])

        # Tolerance
        self.param_widgets['tol'] = QDoubleSpinBox()
        self.param_widgets['tol'].setRange(0.0001, 0.1)
        self.param_widgets['tol'].setSingleStep(0.0001)
        self.param_widgets['tol'].setValue(self.saved_params.get('tol', 0.001))
        self.form_layout.addRow("Tolerance:", self.param_widgets['tol'])

    def save_params(self):
        """Saves the parameters to a dictionary."""
        params = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                value = widget.value()
                # Handle special case for None values (0 with special text)
                if hasattr(widget, 'specialValueText') and widget.specialValueText() and value == 0:
                    params[param_name] = None
                else:
                    params[param_name] = value
            elif isinstance(widget, QDoubleSpinBox):
                params[param_name] = widget.value()
            elif isinstance(widget, QComboBox):
                text = widget.currentText()
                params[param_name] = None if text == 'None' else text
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                params[param_name] = widget.text()

        self.saved_params = params
        self.accept()

    def get_params(self):
        """Returns the saved parameters."""
        return self.saved_params