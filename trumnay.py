from dash import Dash, dcc, html, Input, Output, State
from sklearn import datasets
from sklearn.svm import SVC
import dash_bootstrap_components as dbc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
iris = datasets.load_iris()
 
clf = SVC()
clf.fit(iris.data, iris.target_names[iris.target])
 
X_train,X_test,y_train,y_test=train_test_split(iris.data, iris.target_names[iris.target],test_size=0.33,random_state=42)
 
x_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)
 
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
 
def LabeledSelect(label, **kwargs):
    return dbc.FormGroup([dbc.Label(label), dbc.Select(**kwargs)])
 
# Card components
cards = [
    dbc.Card(
        [
            html.H2(f"{accuracy_score(y_train, x_pred)*100:.2f}%", className="card-title"),
            html.P("Model Training Accuracy", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2(f"{accuracy_score(y_test, y_pred)*100:.2f}%", className="card-title"),
            html.P("Model Test Accuracy", className="card-text"),
        ],
        body=True,
        color="red",
        inverse=True,
    ),
    dbc.Card(
        [
            html.H2(f"{len(X_train)} / {len(X_test)}", className="card-title"),
            html.P("Train / Test Split", className="card-text"),
        ],
        body=True,
        color="black",
        inverse=True,
    ),
]
 
app.layout = html.Div([
    dbc.Row([dbc.Col(card) for card in cards]),
    html.Br(),
    dbc.Input(id="input-on-submit1", placeholder="Input petal length :", type="text"),
    html.Br(),
    dbc.Input(id="input-on-submit2", placeholder="Input petal width :", type="text"),
    html.Br(),
    dbc.Input(id="input-on-submit3", placeholder="Input petal length :", type="text"),
    html.Br(),
    dbc.Input(id="input-on-submit4", placeholder="Input petal width :", type="text"),
    html.Br(),
    dbc.Button("Submit", color="primary", className="d-grid gap-2", id='submit-val', n_clicks=0),
    html.Br(),
    html.Br(),
    html.Div(id='container-button-basic',
             children='Enter a value and press submit')
])
 
 
@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit1', 'value'),
    State('input-on-submit2', 'value'),
    State('input-on-submit3', 'value'),
    State('input-on-submit4', 'value')
)
def update_output(n_clicks, value1, value2, value3, value4):
    test1 = [value1, value2, value3, value4]
   
   
    return 'ผลการทำนาย คือ ::  {} '.format(
        list(clf.predict( [test1] ))
    )
 
if __name__ == "__main__":
    app.run_server(host='127.0.0.1', port='7080')  
