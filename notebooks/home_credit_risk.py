#Python: Notebook Configuration
import sys
from IPython.core.display import display, HTML
import pandas as pd


#PostgreSQL: Setup
from psycopg2 import OperationalError
import psycopg2


#Python: Data Input/Output
import os
from glob import glob #global - The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. Source: https://docs.python.org/3/library/glob.html
import subprocess



#Python: Utility Function
import time



'''Python: Notebook Configuration'''

class Configuration:
    
    train_data_directory_path = '/Users/pauljacob/MyDrive_pauljacob.datascience1@gmail.com/Projects/Home_Credit_Risk/data/raw/train/'
    
    site_packages_directory = "/Users/pauljacob/Library/Python/3.9/lib/python/site-packages"
    
    
    
    
    def __init__(self, train_data_directory_path=None, site_packages_directory=None, drop_tables_from_start=False, skip_create_database_tables=True):
        
        self.drop_tables_from_start = drop_tables_from_start
        self.skip_create_database_tables = skip_create_database_tables
        
        if train_data_directory_path == None:
            self.train_data_directory_path = '/Users/pauljacob/MyDrive_pauljacob.datascience1@gmail.com/Projects/Home_Credit_Risk/data/raw/train/'
        else:
            self.train_data_directory_path = train_data_directory_path
        if site_packages_directory == None:
            self.site_packages_directory = "/Users/pauljacob/Library/Python/3.9/lib/python/site-packages"
        else:
            self.site_packages_directory = site_packages_directory
            
        self.set_the_site_packages_directory()
        self.set_the_notebook_display_settings()
        

    def set_the_notebook_display_settings(self):
        display(HTML("<style>.container { width:99.9% !important; }</style>"))
        pd.options.display.max_columns = 3999
        pd.options.display.max_rows = 999
        pd.set_option('display.max_colwidth', None)
        pd.options.display.max_info_columns = 3999

    def set_the_site_packages_directory(self):
        sys.path.append(self.site_packages_directory)


'''Python: Data Input/Output'''

def get_filename_path_list_from_filename_path_regular_expression(filename_path_regular_expression):
    
    filename_path_list = glob(str(filename_path_regular_expression))
    filename_path_list.sort()
    return filename_path_list


def get_filename_list():
    filename_list = [filename for filename in os.listdir(Configuration.train_data_directory_path) if not filename in ['.DS_Store']]
    filename_list.sort()
    return filename_list




'''PostgreSQL: Setup'''

def generate_parent_paths_from(filename_path):
    components = filename_path.split(os.sep)
    parent_paths = [os.sep.join(components[:i+1]) for i in range(len(components))]
    return '/ '.join(parent_paths[2:])



def create_connection(db_name, db_user, db_password, db_host, db_port):
    '''Source: https://realpython.com/python-sql-libraries/#postgresql'''
    connection = None
    try:
        connection = psycopg2.connect(database=db_name, user=db_user, password=db_password, host=db_host, port=db_port,)
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection


def execute_query(connection, query, table_name=None, column_name_list=None):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query); print("Query executed successfully")
        if "SELECT" in query:
            result=cursor.fetchall()
            cursor.close()
            if table_name != None and column_name_list != None:

                df_list = [pd.DataFrame(result[0+offset:10000+offset]) for offset in range(0, len(result), 10000)]
                df = pd.concat(df_list, axis=0)
                df.columns=column_name_list
                return df
            
            elif table_name != None and column_name_list == None:
                column_name_list=get_the_column_name_list_from_table_name(table_name=table_name, connection=connection)
                df_list = [pd.DataFrame(result[0+offset:10000+offset]) for offset in range(0, len(result), 10000)]
                df = pd.concat(df_list, axis=0)
                return df
            
            elif table_name == None:
                df_list = [pd.DataFrame(result[0+offset:10000+offset]) for offset in range(0, len(result), 10000)]
                if len(df_list) > 0:
                    df=pd.concat(df_list, axis=0)
                    return df
                else:
                    return None
        else:
            cursor.close()
            return None

    except OperationalError as e:
        print(f"The error '{e}' occurred")



def construct_query_from_table_name_column_name_list_and_data_type_list(table_name, column_name_list, data_type_list):

    query = f"CREATE TABLE {table_name} ("

    for column_name, data_type in zip(column_name_list, data_type_list):
        query += f"{column_name} {data_type}, "

    query = query[:-2]; query += ")"
    return query



def create_and_copy_file_to_postgresql_table(connection, table_name, column_name_list, data_type_list, filename_path_list):
    
    #Drop the Table if it Exists
    query = "DROP TABLE IF EXISTS {};".format(table_name)
    execute_query(connection, query=query, table_name=table_name, column_name_list=None)

    #Construct and Execute the Create PostgreSQL Table Query
    query=construct_query_from_table_name_column_name_list_and_data_type_list(table_name=table_name, column_name_list=column_name_list, data_type_list=data_type_list)
    execute_query(connection=connection, query=query, table_name=table_name)

    for filename_path in filename_path_list:
        # Enable Read and Execute Permissions to the .csv File
        filename_parent_paths=generate_parent_paths_from(filename_path=filename_path)
        command = "chmod a+rX {}".format(filename_parent_paths); subprocess.run(command, shell=True)

        #Copy the .csv Data into the Table
        query = "copy {} FROM '{}' DELIMITER ',' CSV HEADER;".format(table_name, filename_path)
        execute_query(connection=connection, query=query, table_name=table_name)
        
        
def extract_database_table_data_type_list_from_dataframe(df):

    dtype_list = list(df.dtypes.reset_index().rename(columns={'index':'name', 0:'values'}).loc[:, 'values'].values)
    column_name_list = list(df.columns)

    database_table_data_type_list = []
    index = 0
    for dtype, column_name in zip(dtype_list, column_name_list):
        if dtype == 'int64':
            database_table_data_type_list += ['INTEGER']
            
        elif (dtype == 'O') and (column_name[-1] in ['D',]):
            database_table_data_type_list += ['DATE']
            
        elif (dtype == 'O') and ('date' in column_name):
            database_table_data_type_list += ['DATE']
            
        elif (dtype == 'O') and (not 'date' in column_name) and (not column_name[-1] in ['D',]):
            database_table_data_type_list += ['VARCHAR']
            
        elif (dtype == 'float64'):
            database_table_data_type_list += ['NUMERIC']
            
        elif (dtype == 'bool'):
            database_table_data_type_list += ['BOOLEAN']
            
    return database_table_data_type_list


def create_database_table_and_return_dataframe_from_single_filename(connection, filename, return_result=True):

    #extract table_name from filename
    table_name = filename.split('.csv')[0]

    #extract data_type_list from .CSV file using pandas
    df = pd.read_csv(Configuration.train_data_directory_path + filename)
    data_type_list = extract_database_table_data_type_list_from_dataframe(df=df)

    #extract column_name_list
    column_name_list = list(df.columns)
    del df

    filename_path_list = [f'{Configuration.train_data_directory_path}{filename}']

    create_and_copy_file_to_postgresql_table(connection=connection, table_name=table_name, column_name_list=column_name_list, data_type_list=data_type_list, filename_path_list=filename_path_list)

    if return_result == False:
        return None
    else:
        #read from the Database Table as a pandas DataFrame
        query=f'SELECT * FROM {table_name}'
        return execute_query(connection=connection, query=query, table_name=table_name, column_name_list=column_name_list)




def create_database_table_and_return_dataframe_from_filename_regular_expression(connection, filename_regular_expression, return_result=True):
    
    #get the filename_path_list
    filename_path_list = get_filename_path_list_from_filename_path_regular_expression(Configuration.train_data_directory_path+filename_regular_expression)


    #extract table_name from filename
    table_name = filename_regular_expression.split('_*.csv')[0]


    #extract data_type_list from .CSV file using pandas
    df_index_0 = pd.read_csv(filename_path_list[0])
    data_type_list = extract_database_table_data_type_list_from_dataframe(df=df_index_0)


    #extract column_name_list
    column_name_list = list(df_index_0.columns)

    del df_index_0

    create_and_copy_file_to_postgresql_table(connection=connection, table_name=table_name, column_name_list=column_name_list, data_type_list=data_type_list, filename_path_list=filename_path_list)

    if return_result == False:
        return None
    else:
        #read from the Database Table as a pandas DataFrame
        query=f'SELECT * FROM {table_name}'
        return execute_query(connection=connection, query=query, table_name=table_name, column_name_list=column_name_list)



'''PostgreSQL: Data Analysis'''
def get_the_column_name_list_from_table_name(table_name, connection,):

    query = "SELECT column_name FROM information_schema.columns WHERE table_name = '{}';".format(table_name)
    cur = connection.cursor()
    cur.execute(query)
    column_name_list = cur.fetchall()
    cur.close()

    return [row[0] for row in column_name_list]

def get_the_table_column_values_and_value_counts(table_name, column_name):
    #Print the Column Name 'approvaldate_319D' Value Counts
    query=\
    "SELECT {}, COUNT({}) AS value_count \
    FROM {} GROUP BY {} ORDER BY value_count DESC;".format(column_name, column_name, table_name, column_name)
    df_value_counts=execute_query(connection, query=query, table_name=table_name, column_name_list=[column_name, 'value_count'])
    print(df_value_counts.head(20))

    #Select the Values of Column Name 'annuity_853A'
    query="SELECT {} FROM {}".format(column_name, table_name)
    return execute_query(connection, query=query, table_name=table_name, column_name_list=[column_name])


def get_the_table_column_data_types_and_value_count(table_name, column_name, data_type):
    #print the Data Type Count of Column Name ' '
    query="SELECT pg_typeof({}) AS {}, COUNT(*) AS count FROM {} GROUP BY {};".format(column_name, data_type, table_name, data_type)
    data_type_value_count=execute_query(connection, query=query, table_name=table_name, column_name_list=[column_name, 'value_count'])
    print(data_type_value_count)
    
    #Get the Data Type of Column Name ' '
    query="SELECT pg_typeof({}) AS {} FROM {};".format(column_name, data_type, table_name)
    return execute_query(connection, query=query, table_name=table_name, column_name_list=[column_name])



'''Python: Utility Function'''

def occupy_the_kernel():
    time.sleep(1000000)

    

'''Python: Data Analysis'''

def preview_df(df):
    """Of this DataFrame, prints the row and column count and then returns the concatenated first 5 and last % rows.

    Args:
        df (DataFrame): This pandas DataFrame object.
    Returns:
        df (DataFrame): The concatenated first 5 and last 5 rows of this pandas DataFrame.
    """
    if df.shape[0] > 9:
        print(df.shape)
        return pd.DataFrame(pd.concat([df.head(5), df.tail(5)]))
    else:
        return df



def preview_list(list_):
    """Print the list length and return the list.
    
    Args:
        list_ (list): The list object to return.
    
    Returns:
        list_ (list): The same list object.
    """
    print(len(list_))
    return list_



def filter_this_data_frame_by_column_name_and_value_list(df, column_name, value_list):
    return df.loc[df.loc[:, column_name].isin(value_list), :]




























#####################################################################################

def plot_vertical_stacked_bar_graph(df, figure_filename, colors, feature_column_name_label, ylabel, xlabel, xtick_dictionary=None, annotation_text_size=11, dpi=100, xtick_rotation=0, annotation_type='frequency', frequency_annotation_round_by_number=-2, y_upper_limit=None, rectangle_annotation_y_offset=None, figsize=None, feature_column_name=None):
    """
    
    Args:
        df (DataFrame): Frequency-percentage DataFrame with index as feature name and values and header as target variable and values
        figure_filename (str): The figure filename.
        colors (list): The two string color list.
        feature_column_name_label (str): feature column name string to use in plot title. 
        ylabel (str): The bar plot ylabel string
        xlabel (str): The bar plot xlabel string
        xtick_dictionary (dict): dictionary mapping x-axis feature values to desired display name string.
        annotation_text_size (int): The annotation text size.
        dpi (int): The dots per inch in saved figure.
        xtick_rotation (int): The degrees of rotation of the xtick labels.
        annotation_type (str): The "frequency" or "percentage" annotation in stacked bar.
        frequency_annotation_round_by_number (int): The number to round the top of bar frequency annotation by.
        y_upper_limit (int): The y-axis upper limit on the bar plot
        rectangle_annotation_y_offset (int): The horizontal offset position for annotation in stacked bar plot.
        figsize (tuple): The x and y dimensions of the figure. Otherwise, None.
        feature_column_name (str): None
    """

    if y_upper_limit == None:
        y_upper_limit = df.loc[:, 'total'].max() * 1.1
    if xtick_rotation==None:
        xtick_rotation = 0
    
    feature_column_name_unique_value_count = df.index.drop_duplicates().shape[0]

    bottom = np.zeros(feature_column_name_unique_value_count)

    
    if min(df.index) == 0:
        index_array = np.arange(feature_column_name_unique_value_count,)
    elif min(df.index) == 1: 
        index_array = [index+1 for index in np.arange(feature_column_name_unique_value_count,)]
    else:
        index_array = np.arange(feature_column_name_unique_value_count,)
        


    if figsize == None: figsize = (8,6)
    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    if y_upper_limit != None:
        axes.set_ylim([0, y_upper_limit])
    for i, target_label_column_name in enumerate(df.loc[:, ['coupon acceptance', 'coupon refusal']].columns):
        axes.bar(df.index, df.loc[:, target_label_column_name], bottom=bottom, label=target_label_column_name.capitalize(), color=colors[i])
        if xtick_dictionary == None:
            axes.set_xticks(index_array, df.index, rotation=xtick_rotation)
        elif xtick_dictionary != None:
            axes.set_xticks(index_array, df.index.map(xtick_dictionary), rotation=xtick_rotation)
        bottom += np.array(df.loc[:, target_label_column_name])

        
    totals = df.loc[:, ['coupon acceptance', 'coupon refusal']].sum(axis=1)
    y_offset = 4
    for i, total in enumerate(totals):
        axes.text(totals.index[i], total + y_offset, round(total, frequency_annotation_round_by_number), ha='center', weight='bold', size=annotation_text_size)

    if rectangle_annotation_y_offset == None:
        rectangle_annotation_y_offset = -35

    if annotation_type == 'frequency':
        for rectangle in axes.patches:
            axes.text(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_height()/2 + rectangle.get_y() + rectangle_annotation_y_offset, round(int(rectangle.get_height()), frequency_annotation_round_by_number), ha='center', color='w', weight='bold', size=annotation_text_size)
    elif annotation_type == 'percentage':
        percentage_list = []
        for column_name in df.loc[:, ['percentage acceptance', 'percentage refusal']].columns:
            percentage_list += df.loc[:, column_name].to_list()
        for rectangle, percentage in zip(axes.patches, percentage_list):
            axes.text(rectangle.get_x() + rectangle.get_width() / 2, rectangle.get_height()/2 + rectangle.get_y() + rectangle_annotation_y_offset, '{:.0f}%'.format(round(percentage, 0)), ha='center', color='w', weight='bold', size=annotation_text_size)

    axes.set_title(str(feature_column_name_label) + ' Frequency Distribution', fontsize=18)
    axes.set_ylabel(ylabel=ylabel, fontsize=17)
    axes.set_xlabel(xlabel=xlabel, fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    axes.legend()
    plt.savefig(figure_filename, bbox_inches='tight', dpi=dpi)

    plt.show()
    

    

    
def get_feature_target_frequency_data_frame(df, feature_column_name='income', target_column_name='Y', append_percentage_true_false=False):
    """Calculate the frequency of feature column per target variable and optionally the percentage of total and return it.
    
    Args:
        df (DataFrame): The DataFrame with feature column name and values.
        feature_column_name (str): The name of the feature column.
        target_column_name (str): The name of the target column.
        append_percentage_true_false (bool): Include percentage of total calculation in the output (True) or not (False).
        
    Returns:
        df (DataFrame): The DataFrame with feature value frequencies per target variable (and percentage of total)
    """

    df = df.value_counts([target_column_name, feature_column_name]).reset_index().pivot(index=feature_column_name, columns=target_column_name).reset_index().droplevel(level=[None,], axis=1).rename(columns={'':feature_column_name, 0:'coupon refusal', 1:'coupon acceptance'}).loc[:, [feature_column_name, 'coupon acceptance', 'coupon refusal']]
    if append_percentage_true_false == False:
        return df
    elif append_percentage_true_false == True:
        df.loc[:, 'total'] = df.loc[:, ['coupon acceptance', 'coupon refusal']].sum(axis=1)
        df.loc[:, 'percentage acceptance'] = df.loc[:, 'coupon acceptance'] / df.loc[:, 'total'] * 100
        df.loc[:, 'percentage refusal'] = df.loc[:, 'coupon refusal'] / df.loc[:, 'total'] * 100
        return df



def sort_data_frame(df, feature_column_name, feature_value_order_list, ascending_true_false=True):
    """Row sort the DataFrame using the feature column name and value order list.
    
    Args:
        df (DataFrame): The DataFrame to be row sorted.
        feature_column_name (str): The column name to sorted on.
        feature_value_order_list (list): The ordered value list to sort by.
        ascending_true_false (bool): The sort by the feature_value_order_list (True) or the reverse (False).
    
    Returns:
        df (DataFrame): The row sorted DataFrame.
    """
    feature_column_name_rank = feature_column_name + '_rank'
    value_order_dictionary = dict(zip(feature_value_order_list, range(len(feature_value_order_list))))
    df.loc[:, feature_column_name_rank] = df.loc[:, feature_column_name].map(value_order_dictionary)
    return df.sort_values([feature_column_name_rank], ascending=ascending_true_false)

