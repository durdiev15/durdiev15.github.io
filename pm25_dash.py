# ==============================================================================
# Open a web browser and navigate to http://127.0.0.1:8050/
# ==============================================================================

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from functools import lru_cache

# --- 1. Data Loading and Caching ---
# Using LRU cache to avoid re-downloading data on every callback,
# which is crucial for a responsive and professional application.

@lru_cache(maxsize=None)
def load_data(url: str) -> pd.DataFrame:
    """Loads a CSV from a URL into a pandas DataFrame and makes index timezone-naive."""
    try:
        df = pd.read_csv(url)
        if 'datetime' in df.columns.str.lower():
            datetime_col = [col for col in df.columns if col.lower() == 'datetime'][0]
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            # FIX: Convert to timezone-naive to prevent UTC offset errors with date picker
            df[datetime_col] = df[datetime_col].dt.tz_localize(None)
            df = df.set_index(datetime_col)
        return df
    except Exception as e:
        print(f"Error loading data from {url}: {e}")
        return pd.DataFrame()

# URLs for the datasets
CLEANED_DATA_URL = 'https://raw.githubusercontent.com/durdiev15/durdiev15.github.io/refs/heads/main/datasets/data_cleaned_final.csv'
IMPUTATION_URL = 'https://raw.githubusercontent.com/durdiev15/durdiev15.github.io/refs/heads/main/datasets/merged_pm25_raw_imputed_data.csv'
ADVANCED_RESULTS_URL = 'https://raw.githubusercontent.com/durdiev15/durdiev15.github.io/refs/heads/main/datasets/ml_advanced_results.csv'
BASELINE_RESULTS_URL = 'https://raw.githubusercontent.com/durdiev15/durdiev15.github.io/refs/heads/main/datasets/ml_baseline_results.csv'
METRICS_URL = 'https://raw.githubusercontent.com/durdiev15/durdiev15.github.io/refs/heads/main/datasets/ml_metrics.csv'

# Pre-process the main cleaned dataset
df_cleaned = load_data(CLEANED_DATA_URL)
COLS_TO_DROP = ['Unnamed: 0', 'pm25_lag_2h', 'pm25_lag_3h', 'pm25_lag_6h', 'pm25_lag_12h', 'pm25_lag_24h']
df_cleaned = df_cleaned.drop(columns=COLS_TO_DROP, errors='ignore')
numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

# --- 2. Dashboard Styling and Configuration ---
# A professional color palette and layout theme.

app = dash.Dash(__name__, title="PM2.5 Forecast for Tashkent", external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], suppress_callback_exceptions=True)
server = app.server

PLOTLY_TEMPLATE = 'plotly_white'
COLORS = {
    'background': '#F9F9F9',
    'text': '#333333',
    'accent': '#007BFF',
    'header': '#FFFFFF',
    'border': '#E0E0E0'
}

# --- 3. Application Layout ---
# A clean, tab-based structure for different analysis sections.

app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'color': COLORS['text'], 'fontFamily': 'Sans-Serif'}, children=[
    # Header
    html.Div(
        style={'backgroundColor': COLORS['header'], 'padding': '20px', 'borderBottom': f'1px solid {COLORS["border"]}'},
        children=[
            html.H1('PM2.5 Air Quality Forecasting: An Interactive Analysis Dashboard', style={'textAlign': 'center', 'margin': '0', 'color': '#2C3E50'}),
            html.H2('Author: Dilshod Durdiev', style={'textAlign': 'center', 'margin': '0', 'fontWeight': '400', 'fontSize': '1.2em', 'color': '#7F8C8D'})
        ]
    ),
    
    # Tabs for different sections
    dcc.Tabs(id="tabs-main", value='tab-eda', children=[
        dcc.Tab(label='Exploratory Data Analysis', value='tab-eda'),
        dcc.Tab(label='Data Imputation Showcase', value='tab-imputation'),
        dcc.Tab(label='Model Performance Evaluation', value='tab-performance'),
    ]),
    
    # Content for each tab will be rendered here
    html.Div(id='tabs-content')
])

# --- 4. Callbacks and Interactive Components ---

@app.callback(Output('tabs-content', 'children'), Input('tabs-main', 'value'))
def render_tab_content(tab):
    """Renders the content for the selected tab."""
    if tab == 'tab-eda':
        return build_eda_tab()
    elif tab == 'tab-imputation':
        return build_imputation_tab()
    elif tab == 'tab-performance':
        return build_performance_tab()

def build_eda_tab():
    """Builds the layout for the Exploratory Data Analysis tab."""
    min_date = df_cleaned.index.min().strftime('%Y-%m-%d')
    max_date = df_cleaned.index.max().strftime('%Y-%m-%d')
    
    return html.Div(style={'padding': '20px'}, children=[
        html.H3('Exploratory Data Analysis on Cleaned PM2.5 Data', style={'textAlign': 'center'}),
        
        # Controls Row
        html.Div(className='row', style={'marginBottom': '20px'}, children=[
            html.Div(className='four columns', children=[
                html.Label('Select Time Series Variable:'),
                dcc.Dropdown(id='eda-ts-variable-dropdown', options=[{'label': i, 'value': i} for i in numeric_cols], value='PM2.5')
            ]),
            html.Div(className='four columns', children=[
                html.Label('Select Scatter Plot Variables (X vs Y):'),
                dcc.Dropdown(id='eda-scatter-x-dropdown', options=[{'label': i, 'value': i} for i in numeric_cols], value='t2m_celsius'),
                dcc.Dropdown(id='eda-scatter-y-dropdown', options=[{'label': i, 'value': i} for i in numeric_cols], value='PM2.5')
            ]),
            html.Div(className='four columns', children=[
                html.Label('Select Date Range:'),
                dcc.DatePickerRange(
                    id='eda-date-picker',
                    min_date_allowed=df_cleaned.index.min(),
                    max_date_allowed=df_cleaned.index.max(),
                    start_date=df_cleaned.index.min(),
                    end_date=df_cleaned.index.max(),
                    display_format='YYYY-MM-DD'
                ),
                html.P(f"Data available from {min_date} to {max_date}", style={'fontSize': '0.8em', 'color': '#7F8C8D'})
            ])
        ]),
        
        # Plots Row 1
        html.Div(className='row', children=[
            html.Div(className='six columns', children=dcc.Graph(id='eda-time-series-plot')),
            html.Div(className='six columns', children=dcc.Graph(id='eda-scatter-plot'))
        ]),

        # Plots Row 2
        html.Div(className='row', style={'marginTop': '20px'}, children=[
             html.Div(className='twelve columns', children=dcc.Graph(id='eda-distribution-plot'))
        ]),

        # Plots Row 3
        html.Div(className='row', style={'marginTop': '20px'}, children=[
            html.Div(className='twelve columns', children=dcc.Graph(id='eda-correlation-heatmap', style={'height': '700px'}))
        ])
    ])

def build_imputation_tab():
    """Builds the layout for the Data Imputation tab."""
    df_imputed = load_data(IMPUTATION_URL)
    min_date = df_imputed.index.min().strftime('%Y-%m-%d')
    max_date = df_imputed.index.max().strftime('%Y-%m-%d')
    
    return html.Div(style={'padding': '20px'}, children=[
        html.H3('Showcase of KNN Imputation on Raw PM2.5 Data', style={'textAlign': 'center'}),
        html.Div(className='row', style={'marginBottom': '20px'}, children=[
            html.Div(className='twelve columns', children=[
                html.Label('Select Date Range to Visualize Imputation:'),
                dcc.DatePickerRange(
                    id='imputation-date-picker',
                    min_date_allowed=df_imputed.index.min(),
                    max_date_allowed=df_imputed.index.max(),
                    start_date=df_imputed.index.min() + pd.Timedelta(days=10),
                    end_date=df_imputed.index.min() + pd.Timedelta(days=20),
                    display_format='YYYY-MM-DD'
                ),
                html.P(f"Data available from {min_date} to {max_date}", style={'fontSize': '0.8em', 'color': '#7F8C8D'})
            ])
        ]),
        dcc.Graph(id='imputation-plot')
    ])

def build_performance_tab():
    """Builds the layout for the Model Performance tab."""
    df_metrics = load_data(METRICS_URL)
    df_results = load_data(BASELINE_RESULTS_URL)
    min_date = df_results.index.min().strftime('%Y-%m-%d')
    max_date = df_results.index.max().strftime('%Y-%m-%d')
    
    baseline_models = ['Linear Regression', 'Random Forest', 'XGBoost']
    advanced_models = ['LSTM', 'Transformer']
    all_models = baseline_models + advanced_models
    
    return html.Div(style={'padding': '20px'}, children=[
        html.H3('Comparative Analysis of Model Performance', style={'textAlign': 'center'}),
        
        dcc.Graph(id='metrics-bar-chart', figure=create_metrics_figure(df_metrics)),
        
        html.Div(className='row', style={'marginTop': '30px'}, children=[
            html.Div(className='four columns', children=[
                html.Label('Select Model to Visualize:'),
                dcc.Dropdown(id='model-select-dropdown', options=[{'label': m, 'value': m} for m in all_models], value='LSTM')
            ]),
            html.Div(className='eight columns', children=[
                html.Label('Select Date Range:'),
                dcc.DatePickerRange(
                    id='results-date-picker',
                    min_date_allowed=df_results.index.min(),
                    max_date_allowed=df_results.index.max(),
                    start_date=df_results.index.min(),
                    end_date=df_results.index.min() + pd.Timedelta(days=10),
                    display_format='YYYY-MM-DD'
                ),
                html.P(f"Data available from {min_date} to {max_date}", style={'fontSize': '0.8em', 'color': '#7F8C8D'})
            ])
        ]),
        dcc.Graph(id='actual-vs-predicted-plot')
    ])

# --- EDA Tab Callbacks ---
@app.callback(
    [Output('eda-time-series-plot', 'figure'),
     Output('eda-distribution-plot', 'figure'),
     Output('eda-correlation-heatmap', 'figure')],
    [Input('eda-ts-variable-dropdown', 'value'),
     Input('eda-date-picker', 'start_date'),
     Input('eda-date-picker', 'end_date')]
)
def update_eda_graphs(ts_var, start_date, end_date):
    dff = df_cleaned.loc[start_date:end_date]

    ts_fig = px.line(dff, x=dff.index, y=ts_var, template=PLOTLY_TEMPLATE, title=f'{ts_var} Over Time')
    ts_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    dist_fig = px.histogram(dff, x=ts_var, nbins=50, template=PLOTLY_TEMPLATE, title=f'Distribution of {ts_var}')
    dist_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    
    corr_matrix = dff[numeric_cols].corr()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        hoverongaps=False,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}"
    ))
    heatmap_fig.update_layout(title='Feature Correlation Matrix', template=PLOTLY_TEMPLATE, margin=dict(l=20, r=20, t=40, b=20))

    return ts_fig, dist_fig, heatmap_fig

@app.callback(
    Output('eda-scatter-plot', 'figure'),
    [Input('eda-scatter-x-dropdown', 'value'),
     Input('eda-scatter-y-dropdown', 'value'),
     Input('eda-date-picker', 'start_date'),
     Input('eda-date-picker', 'end_date')]
)
def update_scatter_plot(scatter_x, scatter_y, start_date, end_date):
    dff = df_cleaned.loc[start_date:end_date]
    
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f'Relationship: {scatter_x} vs. {scatter_y}',
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[{'text': 'No data available for the selected date range.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
        )
        return fig

    if len(dff) > 2000:
        dff_sample = dff.sample(n=2000, random_state=42)
    else:
        dff_sample = dff
        
    scatter_fig = px.scatter(dff_sample, x=scatter_x, y=scatter_y, opacity=0.6, 
                             template=PLOTLY_TEMPLATE, title=f'Relationship: {scatter_x} vs. {scatter_y}')
    scatter_fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return scatter_fig


# --- Imputation Tab Callback ---
@app.callback(
    Output('imputation-plot', 'figure'),
    [Input('imputation-date-picker', 'start_date'),
     Input('imputation-date-picker', 'end_date')]
)
def update_imputation_graph(start_date, end_date):
    df_imputed = load_data(IMPUTATION_URL)
    dff = df_imputed.loc[start_date:end_date]
    
    fig = go.Figure()
    # Plot the imputed data as a dashed line
    fig.add_trace(go.Scatter(
        x=dff.index, 
        y=dff['PM2.5_imputed'], 
        mode='lines', 
        name='After KNN Imputation', 
        line=dict(color='royalblue', width=2, dash='dash')
    ))
    # Plot the original data with gaps
    fig.add_trace(go.Scatter(
        x=dff.index, 
        y=dff['PM2.5_original'], 
        mode='lines', 
        name='Original PM2.5 (with missing)', 
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Comparison of Original vs. KNN Imputed PM2.5 Data',
        xaxis_title='Date',
        yaxis_title='PM2.5 (µg/m³)',
        template=PLOTLY_TEMPLATE,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, orientation="h")
    )
    return fig

# --- Performance Tab Callbacks ---
def create_metrics_figure(df_metrics):
    """Creates the grouped bar chart for model metrics."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=('R-squared (R²)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)'))
    
    df_metrics = df_metrics.sort_values('R2', ascending=False)
    fig.add_trace(go.Bar(x=df_metrics['Model'], y=df_metrics['R2'], name='R²', marker_color=px.colors.qualitative.Vivid), row=1, col=1)
    
    df_metrics = df_metrics.sort_values('RMSE', ascending=True)
    fig.add_trace(go.Bar(x=df_metrics['Model'], y=df_metrics['RMSE'], name='RMSE', marker_color=px.colors.qualitative.Plotly), row=1, col=2)
    
    df_metrics = df_metrics.sort_values('MAE', ascending=True)
    fig.add_trace(go.Bar(x=df_metrics['Model'], y=df_metrics['MAE'], name='MAE', marker_color=px.colors.qualitative.Pastel), row=1, col=3)

    fig.update_layout(
        title_text='Model Performance Metrics Comparison',
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        height=500
    )
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Error (µg/m³)", row=1, col=2)
    fig.update_yaxes(title_text="Error (µg/m³)", row=1, col=3)
    return fig

@app.callback(
    Output('actual-vs-predicted-plot', 'figure'),
    [Input('model-select-dropdown', 'value'),
     Input('results-date-picker', 'start_date'),
     Input('results-date-picker', 'end_date')]
)
def update_results_graph(selected_model, start_date, end_date):
    baseline_models = ['Linear Regression', 'Random Forest', 'XGBoost']
    
    if selected_model in baseline_models:
        df_results = load_data(BASELINE_RESULTS_URL)
        actual_col = f'Actual {selected_model}'
        predicted_col = f'Predicted {selected_model}'
    else:
        df_results = load_data(ADVANCED_RESULTS_URL)
        actual_col = f'Actual {selected_model}'
        predicted_col = f'Predicted {selected_model}'
        
    dff = df_results.loc[start_date:end_date]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dff.index, y=dff[actual_col], mode='lines', name='Actual PM2.5', line=dict(color='black', width=2)))
    fig.add_trace(go.Scatter(x=dff.index, y=dff[predicted_col], mode='lines', name=f'Predicted ({selected_model})', line=dict(color=COLORS['accent'], width=2, dash='dash')))
    
    fig.update_layout(
        title=f'Actual vs. Predicted PM2.5 for {selected_model}',
        xaxis_title='Date',
        yaxis_title='PM2.5 Concentration (µg/m³)',
        template=PLOTLY_TEMPLATE,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    app.run(debug=True)
