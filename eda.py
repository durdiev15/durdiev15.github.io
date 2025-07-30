# main.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import numpy as np

try:
    url = 'https://raw.githubusercontent.com/durdiev15/durdiev15.github.io/refs/heads/main/datasets/clean_automobile_df.csv'
    df = pd.read_csv(url)
    # Remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame() # Create an empty dataframe on error

app = Dash(__name__, title="Automobile EDA Dashboard", external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], suppress_callback_exceptions=True)
server = app.server

# 3. Define App Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Automobile Dataset: Exploratory Data Analysis", style={'textAlign': 'center', 'fontFamily': 'Arial'}),
        html.P("An interactive dashboard to explore vehicle characteristics and pricing.", style={'textAlign': 'center', 'fontFamily': 'Arial'})
    ], style={'padding': '10px'}),

    # Tabs for different analysis sections
    dcc.Tabs(id="tabs-main", value='tab-dist', children=[
        dcc.Tab(label='Distributions', value='tab-dist'),
        dcc.Tab(label='Relationships & Correlation', value='tab-corr'),
        dcc.Tab(label='Price Analysis', value='tab-price'),
        dcc.Tab(label='Categorical Breakdown', value='tab-cat'),
    ]),
    html.Div(id='tabs-content-main', style={'padding': '20px'})
])

# 4. Callback to render tab content
@app.callback(Output('tabs-content-main', 'children'),
              Input('tabs-main', 'value'))
def render_content(tab):
    if df.empty:
        return html.H3("Data could not be loaded. Please check the data source.", style={'color': 'red', 'textAlign': 'center'})

    if tab == 'tab-dist':
        # --- Distribution Tab Content ---
        return html.Div([
            html.H3('Univariate Distributions', style={'textAlign': 'center'}),
            html.Div(className='row', children=[
                html.Div(dcc.Graph(id='hist-price'), className='six columns'),
                html.Div(dcc.Graph(id='hist-horsepower'), className='six columns'),
            ]),
            html.Div(className='row', style={'marginTop': '20px'}, children=[
                html.Div(dcc.Graph(id='hist-city-mpg'), className='six columns'),
                html.Div(dcc.Graph(id='hist-curb-weight'), className='six columns'),
            ]),
            html.Hr(),
            html.H3('Categorical Counts', style={'textAlign': 'center'}),
            html.Div(className='row', children=[
                html.Div(dcc.Graph(id='bar-make'), className='twelve columns'),
            ]),
             html.Div(className='row', style={'marginTop': '20px'}, children=[
                html.Div(dcc.Graph(id='bar-body-style'), className='six columns'),
                html.Div(dcc.Graph(id='bar-drive-wheels'), className='six columns'),
            ]),
        ])
    elif tab == 'tab-corr':
        # --- Correlation Tab Content ---
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[numeric_cols].corr()

        # Create a mask to hide the upper triangle instead of lower triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix_masked = corr_matrix.mask(mask)

        # Create annotations: text only for non-masked entries
        annotations = corr_matrix_masked.round(2).map(lambda x: f'{x:.2f}' if pd.notna(x) else '')

        # Build the heatmap
        heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix_masked.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',  # Red-blue diverging colormap
            zmid=0,
            hoverongaps=False,
            text=annotations.values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))

        # Styling and layout
        heatmap.update_layout(
            title_text='Correlation Matrix of Numerical Features',
            title_x=0.5,
            xaxis_tickangle=-45,
            height=700,
            template='seaborn',
            yaxis_autorange='reversed'  # Flip y-axis for triangle orientation
        )

        # Add grid-like spacing
        heatmap.update_traces(xgap=1, ygap=1)

        # Wrap in Dash layout
        return html.Div([
            html.H3('Relationships Between Features', style={'textAlign': 'center'}),
            dcc.Graph(figure=heatmap),
            html.Hr(),
            html.H3('Scatter Plots', style={'textAlign': 'center'}),
            html.Div(className='row', children=[
                html.Div(dcc.Graph(id='scatter-hp-price'), className='six columns'),
                html.Div(dcc.Graph(id='scatter-engine-price'), className='six columns'),
            ]),
            # Add a new row for the additional scatter plots
            html.Div(className='row', style={'marginTop': '20px'}, children=[
                html.Div(dcc.Graph(id='scatter-curb-weight-price'), className='six columns'),
                html.Div(dcc.Graph(id='scatter-city-mpg-price'), className='six columns'),
            ]),
            # Add a third row for more scatter plots
            html.Div(className='row', style={'marginTop': '20px'}, children=[
                html.Div(dcc.Graph(id='scatter-highway-mpg-price'), className='six columns'),
                html.Div(dcc.Graph(id='scatter-length-price'), className='six columns'),
            ]),
        ])
    elif tab == 'tab-price':
        # --- Price Analysis Tab ---
        return html.Div([
            html.H3('Price Distribution by Categorical Features', style={'textAlign': 'center'}),
            html.Div(className='row', children=[
                 html.Div(dcc.Graph(id='box-price-make'), className='twelve columns'),
            ]),
            html.Div(className='row', style={'marginTop': '20px'}, children=[
                html.Div(dcc.Graph(id='box-price-body'), className='six columns'),
                html.Div(dcc.Graph(id='box-price-cylinders'), className='six columns'),
            ]),
        ])
    elif tab == 'tab-cat':
        # --- Categorical Breakdown Tab ---
        return html.Div([
            html.H3('Hierarchical View of Vehicle Features', style={'textAlign': 'center'}),
            dcc.Graph(id='sunburst-chart', style={'height': '70vh'})
        ])
    return html.Div()


# 5. Callbacks to generate figures
# --- Distribution Figures ---
@app.callback(Output('hist-price', 'figure'), Input('tabs-main', 'value'))
def update_hist_price(tab):
    if tab == 'tab-dist' and not df.empty:
        fig = px.histogram(df, x='price', nbins=30, title='Distribution of Price', marginal='box')
        fig.update_layout(bargap=0.1, template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('hist-horsepower', 'figure'), Input('tabs-main', 'value'))
def update_hist_horsepower(tab):
    if tab == 'tab-dist' and not df.empty:
        fig = px.histogram(df, x='horsepower', nbins=30, title='Distribution of Horsepower', marginal='box')
        fig.update_traces(marker_color='indianred', selector=dict(type='histogram'))
        fig.update_layout(bargap=0.1, template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('hist-city-mpg', 'figure'), Input('tabs-main', 'value'))
def update_hist_city_mpg(tab):
    if tab == 'tab-dist' and not df.empty:
        fig = px.histogram(df, x='city-mpg', nbins=30, title='Distribution of City MPG', marginal='box')
        fig.update_traces(marker_color='seagreen', selector=dict(type='histogram'))
        fig.update_layout(bargap=0.1, template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('hist-curb-weight', 'figure'), Input('tabs-main', 'value'))
def update_hist_curb_weight(tab):
    if tab == 'tab-dist' and not df.empty:
        fig = px.histogram(df, x='curb-weight', nbins=30, title='Distribution of Curb Weight', marginal='box')
        fig.update_traces(marker_color='darkslateblue', selector=dict(type='histogram'))
        fig.update_layout(bargap=0.1, template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('bar-make', 'figure'), Input('tabs-main', 'value'))
def update_bar_make(tab):
    if tab == 'tab-dist' and not df.empty:
        make_counts = df['make'].value_counts().reset_index()
        make_counts.columns = ['make', 'count']
        fig = px.bar(make_counts, x='make', y='count', title='Number of Vehicles by Make', text_auto=True)
        fig.update_layout(template='plotly_white', title_x=0.5)
        fig.update_xaxes(categoryorder='total descending')
        return fig
    return go.Figure()

@app.callback(Output('bar-body-style', 'figure'), Input('tabs-main', 'value'))
def update_bar_body_style(tab):
    if tab == 'tab-dist' and not df.empty:
        body_counts = df['body-style'].value_counts().reset_index()
        body_counts.columns = ['body-style', 'count']
        fig = px.bar(body_counts, x='body-style', y='count', title='Number of Vehicles by Body Style', text_auto=True)
        fig.update_traces(marker_color='goldenrod')
        fig.update_layout(template='plotly_white', title_x=0.5)
        fig.update_xaxes(categoryorder='total descending')
        return fig
    return go.Figure()

@app.callback(Output('bar-drive-wheels', 'figure'), Input('tabs-main', 'value'))
def update_bar_drive_wheels(tab):
    if tab == 'tab-dist' and not df.empty:
        drive_counts = df['drive-wheels'].value_counts().reset_index()
        drive_counts.columns = ['drive-wheels', 'count']
        fig = px.bar(drive_counts, x='drive-wheels', y='count', title='Number of Vehicles by Drive Wheels', text_auto=True)
        fig.update_traces(marker_color='darkcyan')
        fig.update_layout(template='plotly_white', title_x=0.5)
        fig.update_xaxes(categoryorder='total descending')
        return fig
    return go.Figure()

# --- Correlation Figures ---
@app.callback(Output('scatter-hp-price', 'figure'), Input('tabs-main', 'value'))
def update_scatter_hp_price(tab):
    if tab == 'tab-corr' and not df.empty:
        fig = px.scatter(df, x='horsepower', y='price', color='body-style',
                         title='Horsepower vs. Price',
                         hover_data=['make'])
        fig.update_layout(template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('scatter-engine-price', 'figure'), Input('tabs-main', 'value'))
def update_scatter_engine_price(tab):
    if tab == 'tab-corr' and not df.empty:
        fig = px.scatter(df, x='engine-size', y='price', color='drive-wheels',
                         title='Engine Size vs. Price',
                         hover_data=['make'],
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

# Add callbacks for the new scatter plots
@app.callback(Output('scatter-curb-weight-price', 'figure'), Input('tabs-main', 'value'))
def update_scatter_curb_weight_price(tab):
    if tab == 'tab-corr' and not df.empty:
        fig = px.scatter(df, x='curb-weight', y='price', color='drive-wheels',
                         title='Curb Weight vs. Price',
                         hover_data=['make'],
                         color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('scatter-city-mpg-price', 'figure'), Input('tabs-main', 'value'))
def update_scatter_city_mpg_price(tab):
    if tab == 'tab-corr' and not df.empty:
        fig = px.scatter(df, x='city-mpg', y='price', color='fuel-type-diesel',
                         title='City MPG vs. Price',
                         hover_data=['make'])
        fig.update_layout(template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('scatter-highway-mpg-price', 'figure'), Input('tabs-main', 'value'))
def update_scatter_highway_mpg_price(tab):
    if tab == 'tab-corr' and not df.empty:
        fig = px.scatter(df, x='highway-mpg', y='price', color='aspiration-turbo',
                         title='Highway MPG vs. Price',
                         hover_data=['make'])
        fig.update_layout(template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()

@app.callback(Output('scatter-length-price', 'figure'), Input('tabs-main', 'value'))
def update_scatter_length_price(tab):
    if tab == 'tab-corr' and not df.empty:
        fig = px.scatter(df, x='length', y='price', color='body-style',
                         title='Vehicle Length vs. Price',
                         hover_data=['make'],
                         color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(template='plotly_white', title_x=0.5)
        return fig
    return go.Figure()


# --- Price Analysis Figures ---
@app.callback(Output('box-price-make', 'figure'), Input('tabs-main', 'value'))
def update_box_price_make(tab):
    if tab == 'tab-price' and not df.empty:
        fig = px.box(df, x='make', y='price', title='Price Distribution by Make')
        fig.update_layout(template='plotly_white', title_x=0.5)
        fig.update_xaxes(categoryorder='median descending')
        return fig
    return go.Figure()

@app.callback(Output('box-price-body', 'figure'), Input('tabs-main', 'value'))
def update_box_price_body(tab):
    if tab == 'tab-price' and not df.empty:
        fig = px.box(df, x='body-style', y='price', title='Price Distribution by Body Style', color='body-style')
        fig.update_layout(template='plotly_white', showlegend=False, title_x=0.5)
        fig.update_xaxes(categoryorder='median descending')
        return fig
    return go.Figure()

@app.callback(Output('box-price-cylinders', 'figure'), Input('tabs-main', 'value'))
def update_box_price_cylinders(tab):
    if tab == 'tab-price' and not df.empty:
        df_sorted = df.copy()
        df_sorted['num-of-cylinders'] = df_sorted['num-of-cylinders'].astype(str)
        
        fig = px.box(df_sorted, x='num-of-cylinders', y='price', title='Price Distribution by Number of Cylinders', color='num-of-cylinders')
        fig.update_layout(template='plotly_white', showlegend=False, title_x=0.5)
        fig.update_xaxes(categoryorder='median descending')
        return fig
    return go.Figure()

# --- Categorical Breakdown Figure ---
@app.callback(Output('sunburst-chart', 'figure'), Input('tabs-main', 'value'))
def update_sunburst(tab):
    if tab == 'tab-cat' and not df.empty:
        fig = px.sunburst(df, path=['make', 'body-style', 'drive-wheels'],
                          values='price',
                          color='price',
                          hover_data=['horsepower', 'city-mpg'],
                          color_continuous_scale='RdBu',
                          title='Hierarchical Breakdown of Vehicles by Price')
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0), title_x=0.5)
        return fig
    return go.Figure()


# 6. Run the App
if __name__ == '__main__':
    app.run(debug=True)
