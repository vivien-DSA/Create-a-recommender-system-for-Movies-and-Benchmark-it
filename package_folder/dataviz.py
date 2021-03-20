# Function to show summary stats and distribution for a column
def show_pie_box_kde(data, column):
    from matplotlib import pyplot as plt
    import seaborn as sns
    # Create a pie chart of columns counts on the third
    column_count = data[column].value_counts()
    plt.figure(figsize=(5,5))
    plt.pie(column_count)
    plt.title(str(column)+ ' count')
    plt.legend(column_count.keys().tolist(),
              title= str(column),
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    f, (ax_box, ax_hist) = plt.subplots(2, figsize = (15, 5), sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    sns.boxplot(data[column], ax=ax_box)
    sns.distplot(data[column], ax=ax_hist)
    ax_box.set(yticks=[], xlabel = '')
    ax_box.set_title(' Distribution of '+ str(column))
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    plt.show()

    
def show_box_kde(data, column):
    from matplotlib import pyplot as plt
    import seaborn as sns
    f, (ax_box, ax_hist) = plt.subplots(2, figsize = (15, 5), sharex=True, gridspec_kw={"height_ratios": (.20, .80)})

    sns.boxplot(data[column], ax=ax_box)
    sns.distplot(data[column], ax=ax_hist)
    ax_box.set(yticks=[], xlabel = '')
    ax_box.set_title(' Distribution of '+ str(column))
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)
    plt.show()

def show_sparsity(df, col, target):
    from matplotlib import pyplot as plt
    import seaborn as sns
    stats = df[[col, target]].groupby(col).agg(['count', 'mean', 'sum'])
    stats = stats.reset_index()
    stats.columns = [col, 'count', 'mean', 'sum']
    stats_sort = stats['count'].value_counts().reset_index()
    stats_sort = stats_sort.sort_values('index')
    plt.figure(figsize=(15,4))
    plt.bar(stats_sort['index'].astype(str).values[0:20], stats_sort['count'].values[0:20])
    plt.title('Frequency of ' + str(col))
    plt.xlabel('Number frequency')
    plt.ylabel('Frequency')

def show_top20_vs_target(df, col, target):
    '''
    We will use the function show_top20_vs_target which :
    sort the numerical columns and display the top 20 value
    compute the mean of the target and display a purple curve of the evolution of the top 20 value
    '''
    from matplotlib import pyplot as plt
    import seaborn as sns
    stats = df[[col, target]].groupby(col).agg(['count', 'mean', 'sum'])
    stats = stats.reset_index()
    stats.columns = [col, 'count', 'mean', 'sum']
    stats = stats.sort_values('count', ascending=False)
    fig, ax1 = plt.subplots(figsize=(15,4))
    ax2 = ax1.twinx()
    ax1.bar(stats[col].astype(str).values[0:20], stats['count'].values[0:20])
    ax1.set_xticklabels(stats[col].astype(str).values[0:20], rotation='vertical')
    ax2.plot(stats['mean'].values[0:20], color='purple')
    ax2.legend(['Mean of '+str(target)])
    ax2.set_ylim(0,5)
    ax2.set_ylabel('target')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel(col)
    ax1.set_title('Top20 ' + col + 's based on frequency')


