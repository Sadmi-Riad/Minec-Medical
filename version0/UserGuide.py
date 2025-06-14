from PyQt5.QtWidgets import (QDialog, QTextEdit, QVBoxLayout, QPushButton, 
                             QScrollArea, QHBoxLayout)
from PyQt5.QtCore import Qt

def show_user_guide_dialog(parent=None):
    guide_text = """
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11pt;
            color: #333333;
            line-height: 1.5;
            margin: 0;
            padding: 0;
        }
        h1 {
            color: #2c3e50;
            font-size: 20pt;
            margin-bottom: 10px;
            border-bottom: 2px solid  #023C5A;
            padding-bottom: 5px;
        }
        h2 {
            color:  #023C5A;
            font-size: 16pt;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        h3 {
            color:  #023C5A;
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 8px;
        }
        ul, ol {
            padding-left: 25px;
            margin: 8px 0;
        }
        li {
            margin-bottom: 6px;
        }
        b, strong {
            color:  #023C5A;
            font-weight: 600;
        }
        .tip-box {
            background-color: #f8f9fa;
            border-left: 4px solid #023C5A;
            padding: 12px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
        }
        .tip-title {
            color:  #023C5A;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .section {
            margin-bottom: 20px;
        }
        .command {
            font-family: 'Consolas', monospace;
            background-color: #e0e0e0;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .key-combo {
            font-family: 'Arial';
            background-color: #d8e8f8;
            padding: 2px 6px;
            border: 1px solid #d0e3f1;
            border-radius: 4px;
            font-size: 10pt;
        }
        .menu-path {
            color: #023C5A;
            font-weight: 500;
        }
    </style>

    <div class='section'>
        <h1>Minec Toolkit User Guide</h1>
        <p style='font-style: italic; color: #7f8c8d;'>
            Comprehensive reference for all application features and workflows
        </p>
    </div>

    <div class='section'>
        <h2>1. Importation Tab</h2>
        <p>This tab allows you to load, inspect, and prepare your dataset.</p>
        
        <h3>File Operations</h3>
        <ul>
            <li><span class='menu-path'>File → Open:</span> Load a CSV dataset (Ctrl+O)</li>
            <li><span class='menu-path'>File → Save / Save As:</span> Save your dataset (Ctrl+S)</li>
            <li><span class='menu-path'>Edit → Undo:</span> Revert the last action (Ctrl+Z)</li>
            <li><span class='menu-path'>Help → About / User Guide:</span> View app information or help</li>
            <li><span class='menu-path'>View:</span> Minimize, maximize, or close the application</li>
        </ul>
        
        <h3>Dataset Handling</h3>
        <ul>
            <li>After loading, the dataset's <b>name</b>, number of <b>rows</b>, and <b>columns</b> are displayed</li>
            <li>Use checkboxes to <b>Select/Unselect</b> or <b>Remove</b> attributes</li>
            <li>Click on an attribute to view its histogram</li>
            <li>Save histograms via <b>'Save Histogram'</b> button</li>
        </ul>
        
        <div class='tip-box'>
            <div class='tip-title'>PRO TIP</div>
            Double-click attribute names for quick statistics. Use right-click context menu for additional options.
        </div>
    </div>
    
    <div class='section'>
        <h2>2. Preprocessing Tab</h2>
        <p>Tools for cleaning and transforming your dataset.</p>
        <h3>Dataset Overview</h3>
        <ul>
            <li>Displays currently selected attributes</li>
            <li><b>Select All</b> / <b>Unselect All</b> buttons for bulk operations</li>
            <li><b>Operation Message</b> panel shows transformation results</li>
        </ul>
        
        <h3>Transform Section</h3>
        <ul>
            <li><b>Encode Data:</b> Convert between categorical and numerical values</li>
            <li><b>Normalize Data:</b> Scale features to uniform range</li>
            <li><b>Apply:</b> Execute transformation and log results</li>
        </ul>
        
        <h3>Clean Section</h3>
        <ul>
            <li><b>Delete Duplicates:</b> Remove exact or near-duplicate records</li>
            <li><b>Handle Missing Data:</b> Fill using mean/median values</li>
            <li><b>Handle Outliers:</b> Choose between IQR or Isolation Forest methods</li>
            <li><b>Handle Imbalanced Data:</b> Use SMOTE to balance class distribution</li>
        </ul>
        
        <div class='tip-box'>
            <div class='tip-title'>WORKFLOW ADVICE</div>
            <ol>
                <li>Start with transformations before cleaning</li>
                <li>Use undo to revert actions or Use (Ctrl+Z) to revert actions</li>
                <li>Save intermediate versions for traceability</li>
            </ol>
        </div>
    </div>
    
    <div class='section'>
        <h2>3. Estimation Tab</h2>
        <p>Perform regression analysis on your data.</p>
        
        <h3>Validation Methods</h3>
        <ul>
            <li><b>Use Training Set:</b> Apply model to training data</li>
            <li><b>Supplied Test Set:</b> Load external dataset for evaluation</li>
            <li><b>Cross-Validation:</b> Select number of folds (8 or 10 recommended)</li>
            <li><b>Percentage Split:</b> Split dataset into training/testing subsets</li>
        </ul>
        
        <h3>Target Selection</h3>
        <ul>
            <li>Choose dependent variable (target column)</li>
            <li>Visual indicator shows selected target</li>
        </ul>
        
        <h3>Algorithm</h3>
        <ul>
            <li><b>Linear Regression:</b> Model relationships using linear approach</li>
        </ul>
        
        <h3>Output</h3>
        <ul>
            <li><b>Visualization:</b> View resulting linear equation</li>
            <li><b>Estimator Output:</b> Upload/Save models, export predictions</li>
            <li>Detailed statistics including R² score and coefficients</li>
        </ul>
    </div>
    
    <div class='section'>
        <h2>4. Classification Tab</h2>
        <p>Tools for predictive classification models.</p>
        
        <h3>Menu Bar</h3>
        <ul>
            <li><span class='menu-path'>File:</span> Open, Save, Save As</li>
            <li><span class='menu-path'>Edit:</span> Undo, Model Parameters</li>
            <li><span class='menu-path'>Help:</span> About, User Guide</li>
            <li><span class='menu-path'>View:</span> Minimize, Maximize, Close</li>
        </ul>
        
        <h3>Validation Methods</h3>
        <ul>
            <li><b>Use Training Set</b></li>
            <li><b>Supplied Test Set</b></li>
            <li><b>Cross-Validation:</b> 5-fold or 10-fold recommended</li>
            <li><b>Percentage Split:</b> Standard 70/30 or custom ratio</li>
        </ul>
        
        <h3>Target Selection</h3>
        <ul>
            <li>Choose column to predict</li>
            <li>Supports both binary and multiclass problems</li>
        </ul>
        
        <h3>Algorithms</h3>
        <ul>
            <li><b>Logistic Regression:</b> For binary/multiclass classification</li>
            <li><b>Decision Tree:</b> Tree-structured classifier with visualization</li>
        </ul>
        
        <h3>Visualization</h3>
        <ul>
            <li><b>Decision Tree:</b> Interactive tree structure exploration</li>
            <li><b>Confusion Matrix:</b> Detailed performance metrics</li>
            <li><b>ROC Curve:</b> For binary classification tasks</li>
        </ul>
        
        <h3>Classifier Output</h3>
        <ul>
            <li><b>Upload Model:</b> Load saved classifier</li>
            <li><b>Save Model:</b> Export current classifier</li>
            <li><b>Save Result:</b> Export predictions with probabilities</li>
        </ul>
    </div>
    
    <div class='section'>
        <h2>5. Clustering Tab</h2>
        <p>Group similar data points together.</p>
        
        <h3>Menu Bar</h3>
        <ul>
            <li><span class='menu-path'>File:</span> Open, Save, Save As</li>
            <li><span class='menu-path'>Edit:</span> Undo, Model Parameters</li>
            <li><span class='menu-path'>Help:</span> About, User Guide</li>
            <li><span class='menu-path'>View:</span> Minimize, Maximize, Close</li>
        </ul>
        
        <h3>Validation Methods</h3>
        <ul>
            <li><b>Use Training Set</b></li>
            <li><b>Supplied Test Set</b></li>
            <li><b>Percentage Split</b></li>
        </ul>
        
        <h3>Algorithms</h3>
        <ul>
            <li><b>K-Means:</b> Groups data into k clusters</li>
            <li><b>EM Algorithm:</b> Uses probabilities to form clusters</li>
        </ul>
        
        <h3>Configuration</h3>
        <ul>
            <li><b>Number of Clusters:</b> Set optimal k value</li>
            <li><b>Initialization:</b> Random or k-means++</li>
            <li><b>Max Iterations:</b> Control computation time</li>
        </ul>
        
        <h3>Visualization</h3>
        <ul>
            <li><b>Cluster Plot:</b> visualization of clusters</li>
            <li><b>Centroid Markers:</b> Shows cluster centers</li>
            <li><b>Silhouette Plot:</b> Measures cluster quality</li>
        </ul>
        
        <h3>Clusterer Output</h3>
        <ul>
            <li><b>Label Clusters:</b> Based on patterns or majority class</li>
            <li><b>Export Results:</b> Save cluster assignments</li>
        </ul>
    </div>
    
    <div class='section'>
        <h2>6. Feature Selection</h2>
        <p>Identify the most important features in your dataset.</p>
        
        <h3>Workflow</h3>
        <ul>
            <li><b>Target Selection:</b> Choose feature to predict</li>
            <li><b>Automatic Analysis:</b> Runs when target is selected</li>
            <li><b>Feature Importance:</b> Ranked list of features</li>
        </ul>
        
        <h3>Output</h3>
        <ul>
            <li><b>Detailed Steps:</b> Shows preprocessing and model used</li>
            <li><b>Feature Rankings:</b> Importance scores for all features</li>
            <li><b>Recommendations:</b> Suggested features to keep/discard</li>
        </ul>
        
        <div class='tip-box'>
            <div class='tip-title'>BEST PRACTICE</div>
            Always run feature selection before building complex models to improve performance and interpretability.
        </div>
    </div>
    
    
    """

    dialog = QDialog(parent)
    dialog.setWindowTitle("User Guide")
    dialog.setMinimumSize(850, 700)
    
    # Modern styling
    dialog.setStyleSheet("""
        QDialog {
            background-color: #ffffff;
        }
        QPushButton {
            background-color:  #023C5A;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 500;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #023C5A;
        }
        QPushButton:pressed {
            background-color:  #023C5A;
        }
        QScrollArea {
            border: none;
        }
        QTextEdit {
            background-color: white;
            border: none;
            padding: 5px;
        }
    """)

    # Main layout
    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(15, 15, 15, 15)
    main_layout.setSpacing(10)

    # Scrollable content area
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    
    # Text edit with full content
    text_edit = QTextEdit()
    text_edit.setHtml(guide_text)
    text_edit.setReadOnly(True)
    scroll.setWidget(text_edit)
    
    # Button layout
    button_layout = QHBoxLayout()
    button_layout.addStretch()
    
    close_button = QPushButton("Close")
    close_button.setCursor(Qt.PointingHandCursor)
    close_button.clicked.connect(dialog.accept)
    
    # Optional help button
    help_button = QPushButton("Online Help")
    help_button.setCursor(Qt.PointingHandCursor)
    help_button.setStyleSheet("background-color: #95a5a5;")
    help_button.clicked.connect(lambda: webbrowser.open("https://example.com/help"))
    
    button_layout.addWidget(help_button)
    button_layout.addWidget(close_button)

    # Assemble layout
    main_layout.addWidget(scroll)
    main_layout.addLayout(button_layout)

    dialog.setLayout(main_layout)
    dialog.exec_()

# Remember to import webbrowser if using the help button functionality
import webbrowser