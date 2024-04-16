import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import matplotlib.cm as cm


class DataVisualizer:


    """
    A class for visualizing data using different types of plots.
    
    Methods:
    - plot_histogram_with_density_and_mean: Plot a histogram with density and mean line.
    - custom_boxplot: Display a custom boxplot with annotations.
    - calculate_outliers: Calculate and display outliers count and percentage.
    - plot_multi_kde: Plot KDE plots for a numeric column based on a categorical column.
    """
    
    def __init__(self, dataframe):
        """
        Initialize the class with a DataFrame.
        
        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to work with.
        """
        self.dataframe = dataframe

    def plot_histogram_with_density_and_mean(self, column):
        """
        Plot a histogram with KDE overlay and a dashed line indicating the mean,
        with the y-axis representing density. It displays text with mean, median, and mode on the plot.

        Parameters:
        - column: The column to be plotted.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.dataframe[column], kde=True, bins=30, color='purple')
        formatter = FuncFormatter(lambda x, _: f'{x:,.0f}')
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)
        mean_value = self.dataframe[column].mean()
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
        mean = format(self.dataframe[column].mean(), ',.1f')
        median = format(self.dataframe[column].median(), ',.1f')
        mode = format(self.dataframe[column].mode().iloc[0], ',.1f') if not self.dataframe[column].mode().empty else np.NaN
        text_x = mean_value * 7.5  
        text_y = plt.ylim()[1] * 0.9  
        plt.text(text_x, text_y, f'Mean: {mean}\nMedian: {median}\nMode: {mode}', va='top')
        plt.show()

    def custom_boxplot(self, variable, plot_title=None, x_label=None, y_label=None, label_font_size=14, title_font_size=16, x_ticks_font_size=12):
        """
        Display a custom horizontal box plot with annotations for the specified variable from the dataframe.

        Parameters:
        - variable (str): The column in the dataframe for which the boxplot will be generated.
        - plot_title (str, optional): The title for the plot. Default is None.
        - x_label (str, optional): The label for the x-axis. Default is None.
        - y_label (str, optional): The label for the y-axis. Default is None.
        - label_font_size (int, optional): Font size for labels. Default is 14.
        - title_font_size (int, optional): Font size for the plot title. Default is 16.
        - x_ticks_font_size (int, optional): Font size for the x-axis tick labels. Default is 12.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=self.dataframe, x=variable, orient='h', width=0.2, color='lightblue')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_title(plot_title or '', fontsize=title_font_size, color='black', fontname='sans-serif')
        ax.set_xlabel(x_label or '', fontsize=label_font_size, color='black', fontname='sans-serif')
        ax.set_ylabel(y_label or '', fontsize=label_font_size, color='black', fontname='sans-serif')
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        ax.tick_params(axis='x', labelsize=x_ticks_font_size, colors='black')
        text_position = 0.05
        stats = self.dataframe[variable].describe()
        mapping = {'25%': 'Lower Quartile (Q1)', '50%': 'Median (Q2)', '75%': 'Upper Quartile (Q3)', 'mean': 'Mean', 'min': 'Min', 'max': 'Max'}
        for stat, value in stats.items():
            if stat in mapping:
                ax.annotate(f'{mapping[stat]}: {value:,.2f}',
                            xy=(0.95, text_position), 
                            xycoords='axes fraction',
                            fontsize=label_font_size, 
                            ha='right',
                            color='black', 
                            fontname='sans-serif')
                text_position += 0.07
        ax.set_xlim(stats['min'] - 10, stats['max'] + 10)
        plt.show()

    def calculate_outliers(self, column):
        """
        Calculate and display the count and percentage of outliers in a given column.

        Parameters:
        - column (str): The column to analyze for outliers.
        """
        Q1 = self.dataframe[column].quantile(0.25)
        Q3 = self.dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.dataframe[(self.dataframe[column] < lower_bound) | (self.dataframe[column] > upper_bound)]
        total_values = self.dataframe[column].count()
        outliers_count = outliers[column].count()
        outliers_percentage = (outliers_count / total_values) * 100
        print(f"Variable: {column}")
        print(f"Count of outliers: {outliers_count}")
        print(f"Percentage of outliers: {outliers_percentage:.2f}%")
        print(f"Lower bound for outliers: {lower_bound:.2f}")
        print(f"Upper bound for outliers: {upper_bound:.2f}")

    def plot_multi_kde(self, numeric_column, category_column, axis_label_fontsize=10, axis_title_fontsize=12):
        """
        Plot KDE plots for a numeric column based on a categorical column.
        If the categorical column has more than 5 categories, plots for the top 5 most frequent categories are displayed,
        and the total number of categories for that variable is printed below the plot.

        Parameters:
        - numeric_column (str): The name of the numeric column.
        - category_column (str): The name of the categorical column.
        - axis_label_fontsize (int): Font size for the axis labels.
        - axis_title_fontsize (int): Font size for the axis titles.
        """
        
        # Get the top 5 most frequent categories
        top_categories = self.dataframe[category_column].value_counts().nlargest(3).index

        plt.figure(figsize=(12, 7))
        colors = ["#7C51A0", "#00ABA9", "#FF5733", "#33FF57", "#3357FF"]
        
        # Plot KDE for each of the top categories
        for i, category in enumerate(top_categories):
            subset = self.dataframe[self.dataframe[category_column] == category]
            sns.kdeplot(subset[numeric_column], 
                        label=str(category), 
                        fill=True, 
                        color=colors[i % len(colors)])
        
        plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        plt.xlabel(numeric_column, fontsize=axis_title_fontsize, fontname="sans-serif")
        plt.ylabel('Density', fontsize=axis_title_fontsize, fontname="sans-serif")
        plt.xticks(fontname="sans-serif", fontsize=axis_label_fontsize)
        plt.yticks(fontname="sans-serif", fontsize=axis_label_fontsize)
        plt.title(f'KDE Plot of {numeric_column} by {category_column}', fontname="sans-serif")
        plt.legend(title=category_column, title_fontsize=axis_title_fontsize, fontsize=axis_label_fontsize)
        plt.show()
        
        # Print the total number of categories
        total_categories = self.dataframe[category_column].nunique()
        print(f"The '{category_column}' column has a total of {total_categories} categories.")

    
    def plot_multiple_boxplots_matplotlib(self, col_dict, date_column=None, reference_date=None, months_back=None, 
                                          color_palette=cm.Purples, axis_titles=None, plot_titles=None, 
                                          top_n=None, specific_values=None, font_size_axis_titles=14, 
                                          font_size_plot_title=16, font_size_values=10):
        """
        Plots multiple boxplots for the specified numeric columns against categorical columns within a date range.

        Parameters:
        - col_dict (dict): Dictionary where keys are categorical columns and values are the numeric columns to plot.
        - date_column (str, optional): Name of the column with date values used to filter the DataFrame. If None, the full DataFrame is used.
        - reference_date (str or pd.Timestamp, optional): The end date for the time window. Required if date_column is specified.
        - months_back (int, optional): Number of months before the reference_date to filter the DataFrame. Required if date_column is specified.
        - color_palette (matplotlib colormap): Colormap to use for the boxplots.
        - axis_titles (dict, optional): Dictionary with custom axis titles, where keys are column names from col_dict.
        - plot_titles (dict, optional): Dictionary with custom titles for each plot, where keys are column names from col_dict.
        - top_n (dict, optional): Dictionary specifying the number of top categories to plot for each categorical column.
        - specific_values (dict, optional): Dictionary specifying specific categories to plot for each categorical column.
        - font_size_axis_titles (int): Font size for the axis titles.
        - font_size_plot_title (int): Font size for the plot title.
        - font_size_values (int): Font size for the labels on both axes.

        Example of use:
        ---------------
        col_dict = {'PAYOR': 'AMOUNT'}
        plot_multiple_boxplots_matplotlib(df,
                                          col_dict=col_dict,
                                          date_column='RECEIVED_DATE',
                                          reference_date='2023-10-31',
                                          months_back=10,
                                          color_palette=cm.Purples,
                                          top_n={'PAYOR': 5},
                                          font_size_axis_titles=18,
                                          font_size_plot_title=20,
                                          font_size_values=16)
        """
        
        df_filtered = self.dataframe.copy()
        if date_column and reference_date and months_back is not None:
            reference_date = pd.to_datetime(reference_date)
            start_date = reference_date - pd.DateOffset(months=months_back)
            df_filtered = df_filtered[(df_filtered[date_column] >= start_date) & (df_filtered[date_column] <= reference_date)]
        
        for x_col, y_col in col_dict.items():
            if specific_values and x_col in specific_values:
                values_to_plot = specific_values[x_col]
            else:
                n_values = top_n.get(x_col, 10) if top_n else 10
                values_counts = df_filtered[x_col].value_counts().nlargest(n_values)
                values_to_plot = values_counts.index[::-1]  # Reverse to ensure descending order

            data_to_plot = [df_filtered[df_filtered[x_col] == val][y_col].dropna().values for val in values_to_plot]

            plt.figure(figsize=(9, 6))
            boxplots = plt.boxplot(data_to_plot, vert=False, patch_artist=True, labels=values_to_plot)

            colors = color_palette(np.linspace(0.3, 1, len(data_to_plot)))
            for patch, color in zip(boxplots['boxes'], colors):
                patch.set_facecolor(color)

            x_title = axis_titles.get(x_col, x_col) if axis_titles else x_col
            y_title = axis_titles.get(y_col, y_col) if axis_titles else y_col  # Use y_col for the y-axis label

            plot_title = plot_titles.get(x_col, f'{x_col} distribution by {y_col}') if plot_titles else f'{y_col} distribution by {x_col}'

            plt.title(plot_title, fontsize=font_size_plot_title, fontname="sans-serif", color="black")
            plt.ylabel(x_title, fontsize=font_size_axis_titles, fontname="sans-serif", color="black")
            plt.xlabel(y_title, fontsize=font_size_axis_titles, fontname="sans-serif", color="black")
            plt.xticks(fontsize=font_size_values, fontname="sans-serif", color="black", rotation=45, ha='right')
            plt.yticks(fontsize=font_size_values, fontname="sans-serif", color="black")

            plt.gca().xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
            plt.tight_layout()
            plt.show()


    def group_and_count(self, group_column, count_column):
        """
        Group by the specified column and count occurrences of the count_column.
        Also, compute the percentage of each group with respect to the total.
        """
        count_name = f'Count of {count_column}'
        grouped_df = self.dataframe.groupby(group_column)[count_column].count().reset_index(name=count_name)
        
        # Compute the percentage of each group with respect to the total
        total_count = grouped_df[count_name].sum()
        grouped_df['Percentage'] = ((grouped_df[count_name] / total_count) * 100).round(1)
        
        return grouped_df.sort_values(by=count_name, ascending=False)

    def plot_counts(self, grouped_data, x_column, y_column, axis_font_size=10, bar_label_font_size=10):
        """
        Plot the counts of a specified column grouped by another column.
        """
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        
        ax = grouped_data.plot(x=x_column, y=y_column, kind='bar', figsize=(11,7), color='#9370DB')  # Light purple
        plt.title(f'{y_column} by {x_column}', fontsize=axis_font_size + 4)
        plt.ylabel(y_column, fontsize=axis_font_size)
        plt.xlabel(x_column, fontsize=axis_font_size)
        plt.xticks(rotation=45, ha='right', fontsize=axis_font_size)
        plt.yticks(fontsize=axis_font_size)
        
        plt.tight_layout()
        
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        
        for index, value in enumerate(grouped_data[y_column]):
            ax.text(index, value + max(grouped_data[y_column])*0.02, "{:,}".format(value), ha='center', fontsize=bar_label_font_size, color='black')
        
        plt.show()