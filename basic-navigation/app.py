import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shiny import render
from shiny.express import ui, input

import shared
import phase1
import phase2


# Add custom CSS to increase navbar height and nav panel content area
ui.head_content(
    ui.tags.style("""
    /* increase top navbar height */
    .navbar, .navbar-default, .navbar-inverse, .shiny-navbar {
      min-height: 90px !important;
      height: 90px !important;
    }

    /* navbar brand and links vertical alignment */
    .navbar .navbar-brand, .navbar .navbar-nav > li > a,
    .shiny-navbar .navbar-brand, .shiny-navbar .nav > li > a {
      line-height: 90px !important;
      padding-top: 0.5rem !important;
      padding-bottom: 0.5rem !important;
      font-size: 16px !important;
    }

    /* increase content area height and padding */
    .tab-content {
      min-height: 600px !important;
      padding: 2rem !important;
    }

    /* increase navset card height and padding */
    .navset-card {
      min-height: 500px !important;
    }

    .navset-card .card-header, .navset-card .nav {
      padding-top: 1rem !important;
      padding-bottom: 1rem !important;
      min-height: 60px !important;
    }

    /* responsive */
    @media (max-width: 768px) {
      .navbar, .shiny-navbar {
        min-height: 70px !important;
        height: auto !important;
      }
      .navbar .navbar-brand, .navbar .navbar-nav > li > a {
        line-height: 1.4 !important;
      }
    }

    /* Table styling for pandas DataFrames - More specific selectors */
    table.dataframe tbody tr:nth-child(odd) {
        background-color: #f9f9f9 !important;
    }
    table.dataframe tbody tr:nth-child(even) {
        background-color: #ffffff !important;
    }

    /* Alternative approach - target all table rows */
    .shiny-table tbody tr:nth-child(odd) {
        background-color: #f9f9f9 !important;
    }
    .shiny-table tbody tr:nth-child(even) {
        background-color: #ffffff !important;
    }

    /* Style the table headers */
    .dataframe thead th {
        background-color: #e9ecef !important;
        font-weight: bold !important;
    }
    """)
)

ui.page_opts(title="Market Analysis Dashboard")


with ui.nav_panel("Phase 1"):
    with ui.navset_card_underline(title="Penguins Data Analysis"):
        with ui.nav_panel("Plot"):
            print("Phase 1 Plot")

        with ui.nav_panel("Table"):
            print("Phase 1 Table")

with ui.nav_panel("Phase 2"):
    with ui.accordion(id="acc", open="Global Indices - 5 Years Performance"):
        with ui.accordion_panel("Global Indices - 5 Years Performance"):
            with ui.navset_card_underline():
                with ui.nav_panel("Box-Whisker Plot"):
                    @render.plot(height=600)
                    def boxplot():
                        return phase2.boxplot_by_year()
                with ui.nav_panel("Source Code"):
                    ui.tags.pre(
                        ui.code(phase2.boxplot_code())
                    )
            ui.br()
            ui.hr()
            with ui.navset_card_underline():
                with ui.nav_panel("Table of Returns Summary"):
                    @render.table
                    def returns_summary():
                        return phase2.table_of_returns_by_year()
                with ui.nav_panel("Source Code"):
                    ui.tags.pre(
                        ui.code(phase2.table_of_returns_by_year_code())
                        )

        with ui.accordion_panel("Global Indices - Correlation Analysis"):

            # Second card below the first one
            with ui.navset_card_underline():
                with ui.nav_panel("Correlation Matrix"):
                    @render.plot(height=600)
                    def correlation_plot():
                        return phase2.correlation_matrix()

                with ui.nav_panel("Source Code"):
                    ui.tags.pre(
                        ui.code(""" text                      return """)
                    )
            ui.br()
            ui.hr()
            with ui.navset_card_underline():
                with ui.nav_panel("Correlation Matrix 1 year (2024)"):
                    @render.plot(height=600)
                    def correlation_plot_2024():
                        return phase2.correlation_matrix_1_year()

                with ui.nav_panel("Source Code"):
                    ui.tags.pre(
                        ui.code(""" text                      return """)
                    )

with ui.nav_panel("Phase 3"):
    ui.markdown("""
    # Global Indices 5 Years Performance Analytics

    For each index:

    a) **Box-Whisker Plot** of daily returns by "YEAR"
    b) **Table** of daily returns by "YEAR" - (n, mean and std. deviation)
    c) **Bar Plot** of median daily return by "YEAR"
    d) **Heat Map** by "YEAR" and "QUARTER" showing median/mean returns
    """)