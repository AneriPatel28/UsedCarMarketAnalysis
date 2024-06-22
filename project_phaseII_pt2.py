# Install dash_bootstrap_components
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Group
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import plotly.figure_factory as ff
import scipy.stats as stats
import statsmodels.api as sm
import dash_bootstrap_components as dbc
import dash_table
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "Used Car Market Analysis"
url = "https://raw.githubusercontent.com/AneriPatel28/Datavizproject/417b8c552dffab7606ffaf573b94aa26da51d925/CarsData.csv"
data = pd.read_csv(url)

def categorize_by_price(price):
    if price <= 10000:
        return 'Low-Range'
    elif price <= 30000:
        return 'Mid-Range'
    elif price <= 60000:
        return 'Premium'
    elif price <= 100000:
        return 'Luxury'
    else:
        return 'Exotic'

data['Category'] = data['price'].apply(categorize_by_price)

manufacturer_country = {
    'ford': 'United States',
    'volkswagen': 'Germany',
    'vauxhall': 'United Kingdom',
    'merc': 'Germany',
    'BMW': 'Germany',
    'Audi': 'Germany',
    'toyota': 'Japan',
    'skoda': 'Czech Republic',
    'hyundi': 'South Korea'
}
data['Manufacturer_Country'] = data['Manufacturer'].map(manufacturer_country)

data['Value_Ratio'] = data['price'] / data['mileage']

data['year'] = data['year'].astype(object)
def get_market_trends_layout():
    return html.Div([
        html.H1("Market Trend Analysis"),
        dcc.Loading(
            id="loading-market-trends",
            type="default",
            children=[
                html.Div([
                    dcc.RadioItems(
                        id='manufacturer-radio',
                        options=[
                            {'label': 'Ford', 'value': 'ford'},
                            {'label': 'Volkswagen', 'value': 'volkswagen'},
                            {'label': 'Vauxhall', 'value': 'vauxhall'},
                            {'label': 'Mercedes', 'value': 'merc'},
                            {'label': 'BMW', 'value': 'BMW'},
                            {'label': 'Audi', 'value': 'Audi'},
                            {'label': 'Toyota', 'value': 'toyota'},
                            {'label': 'Skoda', 'value': 'skoda'},
                            {'label': 'Hyundai', 'value': 'hyundi'}
                        ],
                        value='toyota',
                        labelStyle={'display': 'inline-block', 'margin-right': '20px'},
                        style={
                            'width': '100%',
                            'display': 'inline-flex',
                            'justify-content': 'space-around',
                            'border': '1px solid #ccc',
                            'padding': '10px',
                            'margin-bottom': '10px',
                            'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                        }
                    )
                ], style={'order': '1'}),
                html.Div(id='graph-container', style={'order': '2', 'margin-bottom': '30px'}),
                html.Div([
                    dcc.RangeSlider(
                        id='year-slider',
                        min=2000,
                        max=2019,
                        value=[2000, 2019],
                        marks={str(year): str(year) for year in range(2000, 2020)},
                        step=1
                    )
                ], style={'order': '3', 'margin-bottom': '10px'}),
                html.Div([
                    html.Button("Back", id="back-button", n_clicks=0, style={'width': '100px'})
                ], style={'display': 'flex', 'justify-content': 'flex-start', 'padding': '20px', 'order': '4'}),
            ]
        )
    ], style={'display': 'flex', 'flex-direction': 'column', 'padding': '20px'})



def generate_heatmap(heatmap_data, selected_manufacturer, selected_years):
    # Create the heatmap with Plotly Express
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Transmission", y="Fuel Type", color="Average Tax"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title=f"Average Tax by Fuel Type and Transmission for {selected_manufacturer} ({selected_years[0]}-{selected_years[1]})",
        aspect="auto",
        color_continuous_scale="Blues_r",
    )

    fig.update_xaxes(showgrid=True, gridcolor='white', side="bottom")
    fig.update_yaxes(showgrid=True, gridcolor='white')

    annotations = []
    for y_index, fuel in enumerate(heatmap_data.index):
        for x_index, trans in enumerate(heatmap_data.columns):
            tax_value = heatmap_data.loc[fuel, trans]

            if not np.isnan(tax_value):
                annotations.append(
                    dict(
                        x=x_index,
                        y=y_index,
                        text=f"{tax_value:.2f}",
                        xref="x",
                        yref="y",
                        showarrow=False,
                        font=dict(size=12, color='white')
                    )
                )

    fig.update_layout(
        annotations=annotations,
        xaxis_title='Transmission Type',
        yaxis_title='Fuel Type',
        coloraxis_colorbar=dict(
            title='Avg Tax (£)',
            titleside='right'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
    )

    return fig


def plot_yearly_price_distribution(data):
    price_data = data.groupby('year')['price'].mean().reset_index()

    price_data['formatted_price'] = price_data['price'].apply(lambda x: f"{x/1000:.2f}k")

    color_scale = px.colors.sequential.Blues

    fig = px.bar(
        price_data,
        x='year',
        y='price',
        title='Yearly Price Distribution',
        labels={'price': 'Average Price (£)'},
        text='formatted_price',
        color='price',
        color_continuous_scale=color_scale,
        color_continuous_midpoint=0.5)


    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(
            title='Year',
            tickmode='linear',
            tick0=price_data['year'].min(),
            dtick=1
        ),
        yaxis=dict(
            title='Average Price (£)',
            tickformat='.2f'
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=80)
    )

    return fig



app.layout = get_market_trends_layout
@app.callback(
    [Output('url', 'pathname'),
     Output('graph-container', 'children')],
    [Input('year-slider', 'value'),
     Input('manufacturer-radio', 'value'),
     Input('back-button', 'n_clicks')]
)
def update_graph(selected_years, selected_manufacturer, n_clicks):

    if n_clicks and n_clicks > 0:
        return '/', dash.no_update


    filtered_data = data[
        (data['year'] >= selected_years[0]) & (data['year'] <= selected_years[1]) &
        (data['Manufacturer'].str.lower() == selected_manufacturer.lower())
    ]

    if filtered_data.empty:
        return dash.no_update, html.Div("No data available for the selected criteria", style={'color': 'red', 'fontSize': 18})
    num_shades = 2
    palette = sns.color_palette("Blues_r", num_shades)
    colors = palette.as_hex()[:]
    trends_data = filtered_data.groupby(['year', 'Manufacturer'])['mileage'].mean().reset_index()
    scatter_trace = go.Scatter(x=trends_data['year'],
        y=trends_data['mileage'],
        mode='lines+markers',
        name=selected_manufacturer,
        line=dict(color=colors[1] , width=3),
        marker=dict(
            size=8,
            line=dict(width=2, color=colors[0] ),
            color=colors[0]
        )
    )

    scatter_layout = go.Layout(
        title=f'Market Trend Analysis: Average Mileage for {selected_manufacturer}',
        xaxis=dict(
            title='Year',
            showgrid=True,
            gridwidth=1,
            gridcolor='LightBlue',
            tickmode='array',
            tickvals=list(range(selected_years[0], selected_years[1] + 1)),
            ticktext=[str(year) for year in range(selected_years[0], selected_years[1] + 1)]
        ),
        yaxis=dict(
            title='Average Mileage',
            showgrid=True,
            gridwidth=1,
            gridcolor='LightBlue'
        ),
        legend=dict(
            title='Manufacturer',
            x=1,
            y=1,
            xanchor='right',
            yanchor='top',
            orientation='v'
        ),
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=40)
    )

    scatter_fig = go.Figure(data=[scatter_trace], layout=scatter_layout)

    category_distribution = filtered_data['Category'].value_counts().reset_index()
    category_distribution.columns = ['Category', 'Counts']

    num_shades = 5
    palette = sns.color_palette("Blues_r", num_shades)
    colors = palette.as_hex()[:]


    donut_trace = go.Pie(
        labels=category_distribution['Category'],
        values=category_distribution['Counts'],
        hole=0.5,
        textposition='outside',
        textinfo='percent+label',
        marker=dict(colors=colors),
    )

    donut_layout = go.Layout(
        title=f'Category Distribution for {selected_manufacturer}',
        margin=dict(l=20, r=20, t=40, b=40),
        width=400,
        height=400
    )

    donut_fig = go.Figure(data=[donut_trace], layout=donut_layout)
    heatmap_data = filtered_data.groupby(['fuelType', 'transmission'])['tax'].mean().unstack().fillna(0)
    heatmap_fig = generate_heatmap(heatmap_data, selected_manufacturer, selected_years)

    price_distribution_fig = plot_yearly_price_distribution(filtered_data)

    graph_layout = html.Div([
        dcc.Loading(
            id="loading-graphs",
            type="circle",
            children=[
                dcc.Graph(figure=scatter_fig, style={'height': '100%'}),
                dcc.Graph(figure=donut_fig, style={'width': '450px', 'height': '450px', 'display': 'inline-block'}),
                dcc.Graph(figure=heatmap_fig, style={'width': '600px', 'height': '500px', 'display': 'inline-block'}),
                dcc.Graph(figure=price_distribution_fig,
                          style={'width': '830px', 'height': '500px', 'display': 'inline-block'})
            ]
        )
    ])


    return dash.no_update, graph_layout


##################################################### MODEL PERFORMANCE ##########################################################################


def get_model_performance_layout():
    return html.Div([
        html.H1("Model Performance Analysis"),
        html.Br(),
        html.Div([
            dcc.Dropdown(
                id='transmission-dropdown',
                options=[
                    {'label': 'Manual', 'value': 'Manual'},
                    {'label': 'Semi-Auto', 'value': 'Semi-Auto'},
                    {'label': 'Automatic', 'value': 'Automatic'},
                    {'label': 'Other', 'value': 'Other'}
                ],
                value='Manual',
                clearable=False,
                style={'width': '300px', 'height': '38px', 'display': 'inline-block'}
            ),
            html.Div([
                html.P("Number of Top Models to Display:",
                       style={'margin': '0 10px 0 0', 'display': 'inline-block', 'vertical-align': 'middle', 'border': '1px solid #ccc', 'padding': '5px'}),
                dcc.Input(
                    id='top-models-input',
                    type='number',
                    value=10,
                    min=1,
                    max=20,
                    style={'width': '100px', 'height': '38px', 'display': 'inline-block'}
                )
            ], style={'display': 'inline-block', 'padding': '5px'}),

            html.Button('Outliers ???', id='toggle-outliers-button', n_clicks=0,
                        style={'height': '38px', 'display': 'inline-block', 'margin-left': '1080px'})
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),

        dcc.Loading(
            id="loading-all-graphs",
            children=html.Div([
                html.Div([
                    dcc.Graph(id='bar-chart', style={'display': 'inline-block', 'width': '50%'}),
                    dcc.Graph(id='box-plot', style={'display': 'inline-block', 'width': '50%'})
                ], style={'display': 'flex', 'width': '100%'}),

                html.Div([
                    dcc.Graph(id='price-histogram', style={'display': 'inline-block', 'width': '33%'}),
                    dcc.Graph(id='pie-chart', style={'display': 'inline-block', 'width': '33%'}),
                    dcc.Graph(id='swarm-chart', style={'display': 'inline-block', 'width': '33%'})
                ])
            ]),
            type="default"
        ),
        # Back button
        html.Div([
        html.A(html.Button('Back', id='back-button', n_clicks=0), href='/')
    ], style={'position': 'absolute', 'bottom': '10px', 'left': '10px'})
    ], style={'display': 'flex', 'flex-direction': 'column', 'padding': '20px'})





def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def remove_outliers_iqr_grouped(data, column):
    grouped_data = data.groupby('Manufacturer_Country')
    filtered_groups = [remove_outliers_iqr(group, column) for _, group in grouped_data]

    filtered_data = pd.concat(filtered_groups)

    return filtered_data


def update_box_plot(filtered_data, remove_outliers):
    filtered_data_copy = filtered_data.copy()
    show_points = not remove_outliers  # If we remove outliers, don't show points
    num_shades = 4
    palette = sns.color_palette("Blues_r", num_shades)
    colors = palette.as_hex()[::2]

    if remove_outliers:
        filtered_data_copy = remove_outliers_iqr_grouped(filtered_data_copy, 'mileage')

    fig = px.box(filtered_data_copy, x='Manufacturer_Country', y='mileage',
                 title='Mileage Distribution by Manufacturer Country',
                 points='outliers' if show_points else False,color_discrete_sequence=colors)
    fig.update_layout(
            title="Mileage Distribution by Manufacturer Country",
            xaxis=dict(
                title='Manufacturer Country',
            ),
            yaxis=dict(
                title='Mileage',
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            template='plotly_white',
        height=400,
        width=800,
        )

    fig.update_xaxes(tickfont=dict(family='serif', size=15, color='black'))

    return fig

def update_manufacturer_swarm_plot(transmission_type):
    if not transmission_type:
        return go.Figure()
    num_shades = 16
    palette = sns.color_palette("Blues", num_shades)
    colors = palette.as_hex()[7::2]
    # Filter the data based on the selected transmission type
    filtered_data = data[data['transmission'] == transmission_type]

    if filtered_data.empty:
        return go.Figure()

    fig = px.strip(filtered_data,
                   y='mileage',
                   x='fuelType',
                   color='fuelType',
                   title='Distribution of Car Prices by Manufacturer for ' + str(transmission_type),
                   template='plotly_white',color_discrete_sequence=colors)

    fig.update_traces(jitter=0.7)
    fig.update_layout(xaxis_title='Mileage', yaxis_title='Manufacturer')
    return fig




@app.callback(
    [Output('bar-chart', 'figure'),
     Output('box-plot', 'figure'),
     Output('price-histogram', 'figure'),
     Output('pie-chart', 'figure'),
     Output('swarm-chart', 'figure')],
    [Input('transmission-dropdown', 'value'),
     Input('top-models-input', 'value'),
     Input('toggle-outliers-button', 'n_clicks')]
)
def update_graph(selected_transmission, top_models, n_clicks):
    if not selected_transmission:
        return go.Figure(), go.Figure(), go.Figure()

    filtered_data = data[data['transmission'] == selected_transmission]
    num_shades = 1
    palette = sns.color_palette("Blues_r", num_shades)
    colors = palette.as_hex()[:]
    price_histogram = px.histogram(
        filtered_data,
        x='price',
        nbins=40,
        histnorm='probability density',
        title=f"Price Distribution for {selected_transmission} Transmission", color_discrete_sequence=colors )
    price_histogram.update_layout(
        xaxis_title="Price",
        yaxis_title="Density",
        bargap=0.1,
        template='plotly_white'
    )

    model_counts = filtered_data.groupby(['model', 'Manufacturer']).size().reset_index(name='count')
    model_counts = model_counts.nlargest(top_models, 'count')
    num_shades = 14
    palette = sns.color_palette("Blues_r", num_shades)
    colors = palette.as_hex()[::3]
    bar_data = []
    for manufacturer in model_counts['Manufacturer'].unique():
        manufacturer_data = model_counts[model_counts['Manufacturer'] == manufacturer]
        bar_trace = go.Bar(
            x=manufacturer_data['model'],
            y=manufacturer_data['count'],
            name=manufacturer
        )
        bar_data.append(bar_trace)

    bar_layout = go.Layout(
        title=f"Top {top_models} Models for {selected_transmission} Transmission",
        xaxis=dict(title='Model'),
        yaxis=dict(title='Count'),
        bargap=0.2,
        height=400,
        width=800,
        template='plotly_white',
        barmode='group'
    )

    bar_fig = go.Figure(data=bar_data, layout=bar_layout)

    for i, trace in enumerate(bar_fig.data):
        trace.marker.color = colors[i]

    bar_fig.update_layout(
        legend_title='Manufacturer',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    bar_fig.update_xaxes(categoryorder='total descending')

    remove_outliers = n_clicks % 2 == 1
    box_fig = update_box_plot(filtered_data, remove_outliers)

    country_counts = filtered_data['Manufacturer_Country'].value_counts()
    country_percentages = (country_counts / country_counts.sum()) * 100


    labels_with_percents = {country: f"{country} {percent:.2f}%" for country, percent in country_percentages.items()}
    filtered_data['label_with_percent'] = filtered_data['Manufacturer_Country'].map(labels_with_percents)
    num_shades = 17
    palette = sns.color_palette("Blues_r", num_shades)
    colors = palette.as_hex()[1:-3:2]
    pie_fig = px.pie(
        filtered_data,
        names='Manufacturer_Country',
        title='Distribution of Cars by Country of Manufacturer',
        color_discrete_sequence=colors
    )

    pie_fig.update_traces(textinfo='percent+label', textposition='outside', hoverinfo='label+percent')

    pie_fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide'
    )


    swarm_fig = update_manufacturer_swarm_plot(selected_transmission)

    return bar_fig, box_fig, price_histogram,pie_fig,swarm_fig



############################################# STATISTICAL ANALYSIS #######################################################################

def get_numerical_columns_options(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    options = [{'label': col, 'value': col} for col in numerical_columns]
    return options
def get_category_dynamics_layout():
    dropdown_options = get_numerical_columns_options(data)

    normality_tests_options = [
        {'label': "Kolmogorov-Smirnov Test", 'value': 'ks'},
        {'label': "D'Agostino's K^2 Test", 'value': 'dagostino'}
    ]
    data_copy = data.copy()
    label_encoders = {}

    categorical_cols = data_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data_copy[col] = label_encoders[col].fit_transform(data[col])

    pca = PCA()
    pca.fit(data_copy)

    singular_values = pca.singular_values_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio) * 100

    components_to_keep = np.argmax(cumulative_explained_variance >= 95) + 1

    n_components_long = range(1, len(cumulative_explained_variance) + 1)


    condition_number = max(singular_values) / min(singular_values)

    rounded_singular_values = np.round(singular_values, 2)
    rounded_explained_variance_ratio = np.round(explained_variance_ratio, 2)
    rounded_cumulative_explained_variance = np.round(cumulative_explained_variance, 2)

    num_features_after_pca = np.count_nonzero(rounded_explained_variance_ratio > 0.0)
    singular_df = pd.DataFrame({
        'Feature': np.arange(1, len(rounded_singular_values) + 1),
        'Singular Value': rounded_singular_values,
        'Explained Variance Ratio': rounded_explained_variance_ratio,
        'Cumulative Explained Variance': rounded_cumulative_explained_variance
    })
    condition_df = pd.DataFrame({'Features': ['Condition Number', 'Num Features After PCA'],
                                 'Values': [condition_number, num_features_after_pca]})

    pca_plot = dcc.Graph(
        id='pca-plot',
        figure={
            'data': [
                {'x': np.arange(1, len(rounded_singular_values) + 1), 'y': rounded_singular_values, 'type': 'bar',
                 'name': 'Singular Values'},
                {'x': np.arange(1, len(rounded_cumulative_explained_variance) + 1),
                 'y': rounded_cumulative_explained_variance, 'type': 'line', 'name': 'Cumulative Explained Variance',
                 'yaxis': 'y2'}
            ],
            'layout': {
                'title': 'Effect of PCA: Singular Values and Cumulative Explained Variance',
                'xaxis': {'title': 'Feature'},
                'yaxis': {'title': 'Singular Value'},
                'yaxis2': {'title': 'Cumulative Explained Variance', 'overlaying': 'y', 'side': 'right'},
                'barmode': 'group'
            }
        }
    )

    pca_table = html.Div([
        dash_table.DataTable(
            id='singular-table',
            columns=[{'name': i, 'id': i} for i in singular_df.columns],
            data=singular_df.to_dict('records'),
            style_table={'overflowX': 'scroll'},
            style_cell={'minWidth': '280px', 'width': '280px', 'maxWidth': '280px', 'overflow': 'hidden',
                        'textOverflow': 'ellipsis'}
        )
    ])
    condition_df = pd.DataFrame({'Features': ['Condition Number', 'Num Features After PCA'],
                                 'Values': [condition_number, num_features_after_pca]})

    condition_table = html.Div([
        dash_table.DataTable(
            id='condition-table',
            columns=[
                {'name': 'Features', 'id': 'Features'},
                {'name': 'Values', 'id': 'Values', 'type': 'numeric',
                 'format': {'specifier': '.2f'}}
            ],
            data=condition_df.to_dict('records'),
            style_table={'overflowX': 'scroll', 'width': '100%'},  # Set width to 50%
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'maxWidth': '180px', 'whiteSpace': 'normal',
                        'overflow': 'hidden', 'textOverflow': 'ellipsis'},
        ),
    ], style={'display': 'inline-block', 'verticalAlign': 'top', 'margin-right': '10px'}) # Set display to inline-block

    download_button= html.Div([
        html.Button("Download PCA Data", id="btn-download-csv"),  # Button to trigger download
        dcc.Download(id="download-csv")  # Component to handle the download
    ], style={'position': 'absolute', 'bottom': '20px', 'right': '20px'})
    back_button = html.Div([
        html.A(html.Button('Back', id='back-button', n_clicks=0), href='/')
    ], style={'position': 'absolute', 'bottom': '20px', 'left': '20px'})
    pca_analysis_content = html.Div([
        html.H1("PCA Analysis"),
        html.Div([
            pca_table,
            html.Div([
                html.H3("Effect of Principal Component Analysis"),
                html.Div([

                    html.Div(id='pca-graph-container', style={'margin-top': '10px'})
                ]),
            ], style={'flex': '1'})
        ], style={'display': 'flex'})
    ])

    return html.Div([
        html.H1("Statistical Analysis"),
        html.Br(),
        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='PCA Analysis', value='tab-pca', children=[
                pca_analysis_content,
                html.Br(),
                condition_table,
                download_button,
                back_button

            ]),
    dcc.Tab(label='Normality Test', value='tab-normality', children=[
                html.Div([
                    html.Br(),
                    html.H3("Select Column to Test for Normality"),

                    dcc.Dropdown(
                        id='normality-test-dropdown',
                        options=dropdown_options,
                        value='price'
                    ),
                    dcc.Graph(id='histogram-plot'),
                    back_button
                ]),
                html.Div([
                    html.Br(),
                    html.H3("Select Normality Tests to Perform"),
                    html.Br(),
                    dcc.Checklist(
                        id='normality-tests-checklist',
                        options=normality_tests_options,
                        value=[]
                    ),
                    html.Br(),
                    html.Div(id='normality-test-results')
                ]),
            ]),
            dcc.Tab(label='Heatmap & Correlation', value='tab-heatmap', children=[
                html.Div([
                    html.Br(),
                    dcc.Slider(
                        id='heatmap-splom-selector',
                        min=0,
                        max=1,
                        step=1,
                        marks={
                            0: {'label': 'Heatmap'},
                            1: {'label': 'Scatter Plot Matrix'}
                        },
                        value=0,
                    ),
                    html.Br(),
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=html.Div(id='heatmap-splom-content')
                    ),
                    back_button
                ], style={'width': '100%', 'height': '100%'})
            ]),
            dcc.Tab(label='Statistical Summary', value='tab-stats', children=[
                html.Div([
                    html.Br(),
                    html.H3("Select Columns for Statistical Summary"),
                    html.Br(),
                    dcc.Checklist(
                        id='statistical-summary-columns',
                        options=[{'label': col.title(), 'value': col} for col in data.columns],
                        value=[],
                        inline=True,
                        style={'margin-bottom': '10px'}
                    ),
                    dcc.Loading(
                        id="loading-statistical-summary",
                        type="default",
                        children=html.Div(id='statistical-summary-results')
                    ),
                    back_button
                ])
            ]),
        ]),
        html.Div(id='tabs-content')
    ])



app.layout = get_category_dynamics_layout()
@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    prevent_initial_call=True
)
def generate_csv(n_clicks):
    data_copy = data.copy()
    label_encoders = {}

    categorical_cols = data_copy.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data_copy[col] = label_encoders[col].fit_transform(data[col])

    pca = PCA()
    pca.fit(data_copy)

    singular_values = pca.singular_values_

    condition_number = max(singular_values) / min(singular_values)

    rounded_singular_values = np.round(singular_values, 2)


    explained_variance_ratio = pca.explained_variance_ratio_
    rounded_explained_variance_ratio = np.round(explained_variance_ratio, 2)

    num_features_after_pca = np.count_nonzero(rounded_explained_variance_ratio > 0.0)
    singular_df = pd.DataFrame({'Feature': np.arange(1, len(rounded_singular_values) + 1),
                                'Singular Value': rounded_singular_values,
                                'Explained Variance Ratio': explained_variance_ratio})

    return dcc.send_data_frame(singular_df.to_csv, filename="my_data.csv")

@app.callback(
    Output('pca-graph-container', 'children'),
    [Input('pca-graph-container', 'id')]
)
def generate_pca_graph(n_clicks):
    data_copy = data.copy()
    label_encoders = {col: LabelEncoder() for col in data_copy.select_dtypes(include=['object']).columns}
    for col, le in label_encoders.items():
        data_copy[col] = le.fit_transform(data_copy[col])

    pca = PCA()
    pca.fit(data_copy.select_dtypes(include=[np.number]))

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio) * 100
    components_to_keep = np.argmax(cumulative_explained_variance >= 95) + 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(1, len(explained_variance_ratio) + 1),
        y=cumulative_explained_variance,
        mode='lines+markers',
        name='Cumulative Explained Variance',
        line=dict(color='blue')
    ))

    fig.add_hline(y=95, line=dict(color='red', dash='dash'), annotation_text="95% Threshold", annotation_position="bottom right")
    fig.add_vline(x=components_to_keep, line=dict(color='green', dash='dash'), annotation_text="Optimal Components: {}".format(components_to_keep), annotation_position="bottom right")

    fig.update_layout(
        title='Cumulative Explained Variance vs Number of Components',
        xaxis_title='Number of Components',
        yaxis_title='Cumulative Explained Variance (%)',
        template='plotly_white'
    )

    return dcc.Graph(figure=fig)

app.layout = get_category_dynamics_layout()

# Layout
app.layout = get_category_dynamics_layout()





@app.callback(
    Output('statistical-summary-results', 'children'),
    [Input('statistical-summary-columns', 'value')]
)
def update_statistical_summary(selected_columns):
    if not selected_columns:
        return html.Div("Select columns to generate statistical summary.")

    summaries = []
    for col in selected_columns:
        summary = data[col].describe().to_frame().reset_index().rename(columns={'index': 'Statistic'})

        if data[col].dtype == 'float64' or data[col].dtype == 'int64':
            # Format numeric columns with .2f precision
            summary[col] = summary[col].apply(lambda x: "{:.2f}".format(x))

        summaries.append(html.Div([
            html.Br(),
            html.H3(f'Statistical Summary for {col} \n'),
            html.Br(),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in summary.columns],
                data=summary.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                fill_width=False,
            ),
            html.Br(),
            html.Div([
                html.H3('Multivariate Kernel Density Estimate \n'),
                dcc.Graph(id=f'density-plot-{col}')
            ])
        ]))

        density_fig = px.density_contour(data, x=col, y=col, title=f'Multivariate KDE for {col}')
        summaries[-1].children[-1].children[1].figure = density_fig

    return summaries

num_shades = 1
palette = sns.color_palette("Blues", num_shades)
colors = palette.as_hex()[::]

# Layout
app.layout = get_category_dynamics_layout()

@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('normality-test-dropdown', 'value')]
)
def update_histogram(selected_column):
    fig = px.histogram(data, x=selected_column,nbins=200 ,title=f'Histogram of {selected_column}',color_discrete_sequence=colors)
    return fig

@app.callback(
    Output('normality-test-results', 'children'),
    [Input('normality-test-dropdown', 'value'),
     Input('normality-tests-checklist', 'value')]
)
def update_normality_tests(selected_column, selected_tests_input):
    if selected_column is None:
        return "No column selected"

    data_sample = data[selected_column].dropna()

    if isinstance(selected_tests_input, list):
        selected_tests = selected_tests_input
    elif isinstance(selected_tests_input, str):
        selected_tests = selected_tests_input.split(',')
    else:
        raise ValueError("selected_tests_input must be a list or a comma-separated string")

    results_text = []

    for test in selected_tests:
        test = test.strip().lower()  # Ensure it is stripped and lowercase
        test_result = f"{test.capitalize()} Test:"
        if test == 'ks':
            ks_stat, ks_p = stats.kstest(data_sample, 'norm', args=(np.mean(data_sample), np.std(data_sample)))
            test_result += f" \n - Test Statistic: {ks_stat:.2f},\n p-value: {ks_p:.2f}"
            test_result += " - \n Result: The data does not appear to follow a normal distribution." if ks_p < 0.05 else " - Result: The data appears to follow a normal distribution."
        elif test == 'dagostino':
            dagostino_stat, dagostino_p = stats.normaltest(data_sample)
            test_result += f" \n - Test Statistic: {dagostino_stat:.2f}, \n p-value: {dagostino_p:.2f}"
            test_result += " \n - Result: The data does not appear to follow a normal distribution." if dagostino_p < 0.05 else " - \n Result: The data appears to follow a normal distribution."

        results_text.append(test_result)

    return "\n\n".join(results_text).strip()
@app.callback(
    Output('heatmap-splom-content', 'children'),
    [Input('heatmap-splom-selector', 'value')]  # Slider sends numeric values now
)
def update_visual(selected_option):
    numerical_data = data.select_dtypes(include=[np.number])

    if selected_option == 0:
        correlation_matrix = numerical_data.corr(method='pearson')
        fig = px.imshow(correlation_matrix, text_auto=".2f", aspect='auto',
                        labels=dict(x='Variable', y='Variable', color='Correlation'),
                        x=correlation_matrix.columns, y=correlation_matrix.columns,
                        color_continuous_scale='Blues', zmin=-1, zmax=1)
        fig.update_layout(title_text='Pearson Correlation Coefficient Heatmap', title_x=0.5, autosize=True)
    elif selected_option == 1:
        fig = px.scatter_matrix(numerical_data,
                                dimensions=numerical_data.columns,
                                title='Scatter Plot Matrix',
                                color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(autosize=True)

    return dcc.Graph(figure=fig, style={'width': '100%', 'height': '100vh'})

def get_home_layout():
    return html.Div([
        html.Div(style={'height': '30px'}),
        html.H1("Car Market Analysis Dashboard", style={'text-align': 'center'}),
        html.Div(style={'height': '40px'}),
        dbc.Row([
            dbc.Col(
                dcc.Link(
                    html.Div([
                        html.Img(src="/assets/tab01.jpg", style={'width': '100%'}),
                        html.P("Market Trend Analysis", style={
                            'text-align': 'center',
                            'font-weight': 'bold',
                            'font-size': '25px',
                            'color': 'black',
                            'font-family': 'Open Sans'}),
                    ], style={'border': '1px solid black', 'padding': '10px'}),
                    href="/market-trends",
                    style={'text-decoration': 'none', 'color': 'black'}
                )
            ),
            dbc.Col(
                dcc.Link(
                    html.Div([
                        html.Img(src="/assets/tab02.jpg", style={'width': '100%'}),
                        html.P("Model Performance Analysis", style={
                            'text-align': 'center',
                            'font-weight': 'bold',
                            'font-size': '25px',
                            'color': 'black',
                            'font-family': 'Open Sans'}),
                    ], style={'border': '1px solid black', 'padding': '10px'}),
                    href="/model-performance",
                    style={'text-decoration': 'none', 'color': 'black'}
                )
            ),
            dbc.Col(
                dcc.Link(
                    html.Div([
                        html.Img(src="/assets/tab03.jpg", style={'width': '100%'}),
                        html.P("Car Industry Statistical Analysis", style={
                            'text-align': 'center',
                            'font-weight': 'bold',
                            'font-size': '25px',
                            'color': 'black',
                            'font-family': 'Open Sans'}),
                    ], style={'border': '1px solid black', 'padding': '10px'}),
                    href="/category-dynamics",
                    style={'text-decoration': 'none', 'color': 'black'}
                )
            ),
        ]),
        html.Div([
            dbc.Button("About Us", id="about-us-button", color="primary",
                       style={'position': 'fixed', 'bottom': '10px', 'left': '10px'}),
            dbc.Tooltip(
                "We analyzed data on used car sales to gain insights into market trends and pricing dynamics.",
                target="about-us-button",
                trigger="hover"
            )
        ]),
        html.Div([
            dbc.Textarea(id="feedback-textarea", placeholder="Enter feedback...", style={
                'width': '180px',
                'height': '90px',
            }),
            dbc.Button("Submit Feedback", id="submit-feedback", color="primary", className="mr-1", style={
                'margin-top': '10px',
            }),
        ], style={
            'position': 'fixed',
            'bottom': '10px',
            'right': '10px',
            'z-index': '1000'
        }),
    ])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/market-trends':
        return get_market_trends_layout()
    elif pathname == '/model-performance':
        return get_model_performance_layout()
    elif pathname == '/category-dynamics':
        return get_category_dynamics_layout()
    else:
        return get_home_layout()


@app.callback(
    Output('feedback-textarea', 'value'),
    [Input('submit-feedback', 'n_clicks')],
    [State('feedback-textarea', 'value')],
    prevent_initial_call=True
)
def submit_feedback(n_clicks, feedback):
    if n_clicks and feedback:

        return ""
    else:
        return dash.no_update

app.server.run(
    debug=False,
    port=8080,
    host='0.0.0.0'
)