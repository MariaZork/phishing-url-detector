# -*- coding: utf-8 -*-
import validators
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from datetime import datetime

from inference import Inference


PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

FOOTER_STYLE = {
    "position": "fixed",
    "bottom": 0,
    "left": 0,
    "right": 0,
    "height": "5rem",
    "padding": "1rem",
    "align": "center",
    "background-color": "#343a40",
    "font-weight": "bold",
    "color": "white",
    "text-align": "center"
}

INPUT_STYLE = {
    "display": "flex",
    "justifyContent": "center",
    "padding-top": "15rem",
    "padding-left": "15rem",
    "padding-right": "15rem"
}

BUTTON_STYLE = {
    "display": "flex",
    "justifyContent": "center",
    "padding-top": "2rem",
}

OUTPUT_STYLE = {
    "display": "flex",
    "justifyContent": "center",
    "padding-top": "5rem",
}

navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Phishing URL Detector using Random Forest Algorithm", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://mariazork.github.io/machine%20learning/2021/07/29/phishing-url-detection.html",
        ),
    ],
    id='navbar',
    color="dark",
    dark=True,
    expand='md'
)

header = html.Div(
    [
        navbar
    ]
)

footer = html.Div(
    [
        html.P(f"Copyright: Built by Maria Zorkaltseva, {datetime.now().year}")
    ],
    style=FOOTER_STYLE
)

content = html.Content(
    [
        html.Div(
            children=[dbc.Input(id="url-input",
                                placeholder="Input your URL...",
                                type="text",
                                valid=False,
                                invalid=False)],
            style=INPUT_STYLE
        ),

        html.Div(
            children=[dbc.Button("Reset",
                                 id="reset-button",
                                 color="danger",
                                 className="mr-1",
                                 n_clicks=0),
                      dbc.Button("Predict",
                                 id="predict-button",
                                 color="primary",
                                 className="mr-1",
                                 disabled=False,
                                 n_clicks=0)
                      ],
            style=BUTTON_STYLE
        ),

        html.Div(id='prediction-output', style=OUTPUT_STYLE)
    ]
)


app.layout = html.Div(
    [
        header,
        html.Br(),
        content,
        footer
    ]
)


def is_valid_url(url_value):
    return validators.url(url_value)


@app.callback(
    [
    Output(component_id='url-input', component_property='valid'),
    Output(component_id='url-input', component_property='invalid'),
    Output(component_id='predict-button', component_property='disabled'),
    ],
    Input(component_id='url-input', component_property='value')
)
def sanity_check(url_value):
    if not url_value:
        return False, False, False

    if is_valid_url(url_value):
        return True, False, False
    else:
        return False, True, True


@app.callback(
    Output(component_id='prediction-output', component_property='children'),
    Input(component_id='predict-button', component_property='n_clicks'),
    State(component_id='url-input', component_property='value'),
)
def predict(num_clicks, url_value):
    if url_value:
        predicted_label = Inference.infer(sample=url_value,
                                          model_filename="models/model_0.pkl",
                                          vectorizer_filename="models/tf_idf_0.pkl",
                                          scaler_filename="models/scaler_0.pkl")

        return 'Prediction Label: {}'.format(predicted_label)
    else:
        return " "

@app.callback(
    [Output(component_id='url-input', component_property='value'),
    Output(component_id='predict-button', component_property='n_clicks')],
    Input(component_id='reset-button', component_property='n_clicks'),
)
def reset(num_clicks):
        return None, None

# @app.callback(
#     [Output(component_id='prediction-output', component_property='children'),
#      Output(component_id='url-input', component_property='valid'),
#      Output(component_id='url-input', component_property='invalid')],
#     Input(component_id='predict-button', component_property='n_clicks'),
#     State(component_id='url-input', component_property='value'),
# )
# def predict(num_clicks, url_value):
#     if not url_value:
#         return dash.no_update, dash.no_update, dash.no_update
#
#     if is_valid_url(url_value):
#         predicted_label = Inference.infer(sample=url_value,
#                                           model_filename="models/model_0.pkl",
#                                           vectorizer_filename="models/tf_idf_0.pkl",
#                                           scaler_filename="models/scaler_0.pkl")
#
#         return 'Prediction Label: {}'.format(predicted_label), True, False
#     else:
#         return " ", False, True


if __name__ == '__main__':
    app.run_server(debug=True)