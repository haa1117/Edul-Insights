import dash
from dash import dcc, html, Input, Output, dash_table, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from models import EduAnalyticsML

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "EduInsights - Academic Analytics Dashboard"

# Global variables
ml_system = None

def load_data():
    """Load data from database"""
    global ml_system
    try:
        conn = sqlite3.connect('eduinsights.db')
        students_df = pd.read_sql_query("SELECT * FROM students", conn)
        performance_df = pd.read_sql_query("SELECT * FROM performance", conn)
        attendance_df = pd.read_sql_query("SELECT * FROM attendance", conn)
        teachers_df = pd.read_sql_query("SELECT * FROM teachers", conn)
        conn.close()
        
        # Initialize ML system
        ml_system = EduAnalyticsML()
        ml_system.load_data()
        ml_system.prepare_features()
        
        # Try to load existing models or train new ones
        try:
            ml_system.load_models()
        except:
            print("Training new models...")
            ml_system.train_risk_model()
            ml_system.train_performance_model()
            ml_system.save_models()
        
        return students_df, performance_df, attendance_df, teachers_df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load initial data
students_df, performance_df, attendance_df, teachers_df = load_data()

# Convert dates
if not performance_df.empty:
    performance_df['assessment_date'] = pd.to_datetime(performance_df['assessment_date'])
if not attendance_df.empty:
    attendance_df['date'] = pd.to_datetime(attendance_df['date'])

# Define color scheme
colors = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="#", id="nav-dashboard")),
        dbc.NavItem(dbc.NavLink("Students", href="#", id="nav-students")),
        dbc.NavItem(dbc.NavLink("Performance", href="#", id="nav-performance")),
        dbc.NavItem(dbc.NavLink("Analytics", href="#", id="nav-analytics")),
        dbc.NavItem(dbc.NavLink("Reports", href="#", id="nav-reports")),
    ],
    brand="🎓 EduInsights",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4"
)

# Key metrics cards
def create_metric_card(title, value, icon, color="primary", subtitle=""):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.I(className=f"fas fa-{icon} fa-2x", style={'color': colors[color]}),
                ], className="col-3 d-flex align-items-center justify-content-center"),
                html.Div([
                    html.H2(value, className="mb-0", style={'color': colors[color]}),
                    html.P(title, className="mb-0 text-muted"),
                    html.Small(subtitle, className="text-muted") if subtitle else None
                ], className="col-9")
            ], className="row")
        ])
    ], className="mb-3", style={'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'})

# Sidebar
sidebar = html.Div([
    html.H4("Quick Actions", className="mb-3"),
    dbc.Nav([
        dbc.NavLink([
            html.I(className="fas fa-tachometer-alt me-2"),
            "Overview"
        ], href="#", id="sidebar-overview", active=True),
        dbc.NavLink([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "At-Risk Students"
        ], href="#", id="sidebar-risk"),
        dbc.NavLink([
            html.I(className="fas fa-chart-line me-2"),
            "Performance Trends"
        ], href="#", id="sidebar-trends"),
        dbc.NavLink([
            html.I(className="fas fa-user-graduate me-2"),
            "Student Profiles"
        ], href="#", id="sidebar-profiles"),
        dbc.NavLink([
            html.I(className="fas fa-clipboard-list me-2"),
            "Interventions"
        ], href="#", id="sidebar-interventions"),
    ], vertical=True, pills=True),
    html.Hr(),
    html.H5("Filters", className="mb-3"),
    html.Div([
        html.Label("Grade Level"),
        dcc.Dropdown(
            id='grade-filter',
            options=[{'label': f'{i}th Grade', 'value': f'{i}th'} for i in range(9, 13)],
            value=None,
            multi=True,
            placeholder="All Grades"
        ),
    ], className="mb-3"),
    html.Div([
        html.Label("Subject"),
        dcc.Dropdown(
            id='subject-filter',
            options=[{'label': s, 'value': s} for s in ['Mathematics', 'Science', 'English', 'History', 'Geography', 'Physics', 'Chemistry', 'Biology']],
            value=None,
            multi=True,
            placeholder="All Subjects"
        ),
    ], className="mb-3"),
], className="p-3", style={'background-color': colors['light'], 'min-height': '100vh'})

# Main content area
def get_overview_content():
    if ml_system and hasattr(ml_system, 'features_df'):
        summary = ml_system.get_analytics_summary()
        
        metrics_row = dbc.Row([
            dbc.Col([
                create_metric_card(
                    "Total Students", 
                    summary['total_students'], 
                    "users", 
                    "primary"
                )
            ], width=3),
            dbc.Col([
                create_metric_card(
                    "At-Risk Students", 
                    summary['at_risk_count'], 
                    "exclamation-triangle", 
                    "danger",
                    f"{summary['at_risk_percentage']}% of total"
                )
            ], width=3),
            dbc.Col([
                create_metric_card(
                    "Avg Performance", 
                    f"{summary['avg_performance']}%", 
                    "chart-line", 
                    "success"
                )
            ], width=3),
            dbc.Col([
                create_metric_card(
                    "Avg Attendance", 
                    f"{summary['avg_attendance']}%", 
                    "calendar-check", 
                    "warning"
                )
            ], width=3),
        ], className="mb-4")
        
        # Charts row
        charts_row = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="performance-distribution")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Grade-wise Analytics"),
                    dbc.CardBody([
                        dcc.Graph(id="grade-analytics")
                    ])
                ])
            ], width=6),
        ], className="mb-4")
        
        return html.Div([metrics_row, charts_row])
    else:
        return html.Div([
            dbc.Alert("Data loading in progress. Please wait...", color="info")
        ])

def get_risk_students_content():
    if ml_system and hasattr(ml_system, 'features_df'):
        at_risk_students = ml_system.features_df[ml_system.features_df['at_risk'] == 1]
        
        return html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("At-Risk Students", className="mb-0"),
                    dbc.Badge(f"{len(at_risk_students)} students", color="danger", className="ms-2")
                ]),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='risk-students-table',
                        columns=[
                            {'name': 'Student ID', 'id': 'student_id'},
                            {'name': 'Performance', 'id': 'avg_performance', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                            {'name': 'Attendance', 'id': 'attendance_rate', 'type': 'numeric', 'format': {'specifier': '.1%'}},
                            {'name': 'Risk Level', 'id': 'risk_level'},
                        ],
                        data=at_risk_students.to_dict('records'),
                        style_cell={'textAlign': 'left'},
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        page_size=10,
                        sort_action="native"
                    )
                ])
            ])
        ])
    else:
        return dbc.Alert("Data not available", color="warning")

# App layout
app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col([sidebar], width=3),
        dbc.Col([
            html.Div(id="main-content", children=get_overview_content())
        ], width=9)
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output('main-content', 'children'),
    [Input('sidebar-overview', 'n_clicks'),
     Input('sidebar-risk', 'n_clicks'),
     Input('sidebar-trends', 'n_clicks'),
     Input('sidebar-profiles', 'n_clicks'),
     Input('sidebar-interventions', 'n_clicks')]
)
def update_main_content(overview_clicks, risk_clicks, trends_clicks, profiles_clicks, interventions_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return get_overview_content()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'sidebar-overview':
        return get_overview_content()
    elif button_id == 'sidebar-risk':
        return get_risk_students_content()
    elif button_id == 'sidebar-trends':
        return get_trends_content()
    elif button_id == 'sidebar-profiles':
        return get_profiles_content()
    elif button_id == 'sidebar-interventions':
        return get_interventions_content()
    
    return get_overview_content()

@app.callback(
    Output('performance-distribution', 'figure'),
    [Input('grade-filter', 'value'),
     Input('subject-filter', 'value')]
)
def update_performance_distribution(selected_grades, selected_subjects):
    if performance_df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    filtered_df = performance_df.copy()
    
    if selected_grades:
        # Filter by grades - need to join with students data
        grade_students = students_df[students_df['grade'].isin(selected_grades)]['student_id']
        filtered_df = filtered_df[filtered_df['student_id'].isin(grade_students)]
    
    if selected_subjects:
        filtered_df = filtered_df[filtered_df['subject'].isin(selected_subjects)]
    
    fig = px.histogram(
        filtered_df, 
        x='percentage', 
        bins=20,
        title="Performance Distribution",
        labels={'percentage': 'Performance (%)', 'count': 'Number of Students'},
        color_discrete_sequence=[colors['primary']]
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

@app.callback(
    Output('grade-analytics', 'figure'),
    [Input('subject-filter', 'value')]
)
def update_grade_analytics(selected_subjects):
    if performance_df.empty or students_df.empty:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)
    
    # Merge performance with student data to get grades
    merged_df = performance_df.merge(students_df[['student_id', 'grade']], on='student_id')
    
    if selected_subjects:
        merged_df = merged_df[merged_df['subject'].isin(selected_subjects)]
    
    grade_performance = merged_df.groupby('grade')['percentage'].mean().reset_index()
    
    fig = px.bar(
        grade_performance,
        x='grade',
        y='percentage',
        title="Average Performance by Grade",
        labels={'percentage': 'Average Performance (%)', 'grade': 'Grade Level'},
        color='percentage',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def get_trends_content():
    if performance_df.empty:
        return dbc.Alert("No performance data available", color="warning")
    
    # Performance trends over time
    performance_trend = performance_df.groupby(performance_df['assessment_date'].dt.to_period('M'))['percentage'].mean().reset_index()
    performance_trend['assessment_date'] = performance_trend['assessment_date'].astype(str)
    
    trend_fig = px.line(
        performance_trend,
        x='assessment_date',
        y='percentage',
        title="Performance Trends Over Time",
        labels={'percentage': 'Average Performance (%)', 'assessment_date': 'Month'}
    )
    
    return dbc.Card([
        dbc.CardHeader("Performance Trends Analysis"),
        dbc.CardBody([
            dcc.Graph(figure=trend_fig)
        ])
    ])

def get_profiles_content():
    if students_df.empty:
        return dbc.Alert("No student data available", color="warning")
    
    return dbc.Card([
        dbc.CardHeader("Student Profiles"),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[
                    {'name': 'Student ID', 'id': 'student_id'},
                    {'name': 'Name', 'id': 'name'},
                    {'name': 'Grade', 'id': 'grade'},
                    {'name': 'Gender', 'id': 'gender'},
                    {'name': 'Family Income', 'id': 'family_income'},
                ],
                data=students_df.head(20).to_dict('records'),
                style_cell={'textAlign': 'left'},
                page_size=10,
                sort_action="native",
                filter_action="native"
            )
        ])
    ])

def get_interventions_content():
    if not ml_system or not hasattr(ml_system, 'features_df'):
        return dbc.Alert("ML system not available", color="warning")
    
    intervention_students = ml_system.features_df[ml_system.features_df['needs_intervention'] == 1]
    
    interventions_data = []
    for _, student in intervention_students.head(10).iterrows():
        recommendations = ml_system.get_intervention_recommendations(student['student_id'])
        interventions_data.append({
            'student_id': student['student_id'],
            'recommendations': '; '.join(recommendations)
        })
    
    return dbc.Card([
        dbc.CardHeader([
            "Intervention Recommendations",
            dbc.Badge(f"{len(intervention_students)} students", color="warning", className="ms-2")
        ]),
        dbc.CardBody([
            dash_table.DataTable(
                columns=[
                    {'name': 'Student ID', 'id': 'student_id'},
                    {'name': 'Recommended Interventions', 'id': 'recommendations'},
                ],
                data=interventions_data,
                style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
                style_cell_conditional=[
                    {'if': {'column_id': 'recommendations'}, 'width': '70%'}
                ],
                page_size=10
            )
        ])
    ])

if __name__ == '__main__':
    app.run(debug=True, port=8050) 