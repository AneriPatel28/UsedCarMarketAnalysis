# Name: Aneri Patel
# GWID: G40408020

#Project Phase 01:

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import scipy.stats as stats
from io import StringIO
import prettytable
from matplotlib.ticker import FuncFormatter

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
sns.set_style("darkgrid")
#%%
pd.set_option('display.max_columns', None)
url = "https://raw.githubusercontent.com/AneriPatel28/Datavizproject/417b8c552dffab7606ffaf573b94aa26da51d925/CarsData.csv"
data = pd.read_csv(url)
print(data.info())

#%%
print(data['price'].nunique())
print(data['price'].unique())
print(data.shape)
#%%

print(data['price'].min())
print(data['price'].max())

#%%

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

data.head()
#%%

data['Category'].value_counts()

#%%
data['Manufacturer'].value_counts()
#%%
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
print(data['Manufacturer_Country'].value_counts())
#%%

data['Value_Ratio'] = data['price'] / data['mileage']

#%%
print(data.info())
#%%
print(data.head(3))
print(data['engineSize'].value_counts())

#%%

data['year'] = data['year'].astype(object)

print(data.info())
#%%
print(data['year'].value_counts())


buffer = StringIO()
data.info(buf=buffer)
info_str = buffer.getvalue()

# Create a PrettyTable
table = prettytable.PrettyTable()
table.field_names = ["Info"]

# Split the output string into lines for processing
info_lines = info_str.split('\n')

# Add each line as a separate row in the table
for line in info_lines:
    if line:  # avoid adding empty lines
        table.add_row([line.strip()])

# Print the table
print(table)

#%%



#%%




#########################################################################################

#%%
print(data['transmission'].value_counts())
#%%

#### Count Plot
plt.figure(figsize=(10, 7))
# Count plot for the 'model' column
plot=sns.countplot(x='Manufacturer', data=data, order = data['Manufacturer'].value_counts().index,palette='YlGnBu')

plt.title('Number of Cars by Manufacturer', fontdict={'fontsize': 23, 'fontname': 'serif', 'color': 'blue'})
plt.xlabel('Manufacturer', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Number of Cars', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.grid(True, color='grey', linestyle='-', linewidth=0.5, alpha=0.25)

plt.xticks(fontname='serif')
plt.yticks(fontname='serif')
plt.show()


#%%

# Heatmap with cbar

grouped = data.groupby(['transmission', 'fuelType']).size().reset_index(name='count')

pivot_table = grouped.pivot(index='transmission', columns='fuelType', values='count')
pivot_table=pivot_table.fillna(0)
def custom_format(x):
    if x >= 1000:
        return '{:.2f} K'.format(x/1000)
    else:
        return str(x)
annot_array = np.vectorize(custom_format)(pivot_table.values)
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=annot_array, cmap='YlGnBu', fmt='s', cbar=True, linewidths=0.03, linecolor='grey')
plt.title('Count of Cars by Transmission and Fuel Type', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'blue'})
plt.xlabel('Fuel Type', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Transmission Type', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.xticks(fontname='serif', fontsize=12)
plt.yticks(fontname='serif', fontsize=12)
plt.show()


#%%



#%%

# BarPlot with subplots:

data_group = data.groupby(['Category', 'model']).agg({'mileage': 'mean'}).reset_index()

top_models_by_category = data_group.groupby('Category').apply(lambda x: x.nlargest(7, 'mileage')).reset_index(drop=True)

categories = top_models_by_category['Category'].unique()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
axs = axs.flatten()

bottom_row_indices = [len(axs) - 2, len(axs) - 1]
first_column_indices = [0, len(axs) // 2]

for i, (ax, category) in enumerate(zip(axs, categories)):
    category_data = top_models_by_category[top_models_by_category['Category'] == category]
    sns.barplot(data=category_data, x='model', y='mileage', ax=ax, palette='YlGnBu')
    ax.set_title(category, fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'blue'})

    if i in bottom_row_indices:
        ax.set_xlabel('Model', fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})
    else:
        ax.set_xlabel('')

    if i in first_column_indices:
        ax.set_ylabel('Avg. Mileage', fontdict={'fontsize': 14, 'fontname': 'serif', 'color': 'darkred'})
    else:
        ax.set_ylabel('')

    ax.set_xticklabels(ax.get_xticklabels(), fontname='serif', fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), fontname='serif', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.suptitle('Top 7 Models by Mileage Across Different Categories', fontsize=24, fontname='serif', color='blue',
             va='bottom', y=0.93)
plt.show()

#%%

# Pie Charts
country_counts = data['Manufacturer_Country'].value_counts()
print(country_counts)

num_shades = 8
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()
print(colors)
plt.figure(figsize=(8,7))
pies, texts, autotexts = plt.pie(country_counts, labels=country_counts.index, autopct='%1.2f%%', startangle=140, colors=colors)
plt.title('Distribution of Cars by Manufacturer Country', fontdict={'fontsize': 20, 'fontname': 'serif', 'color': 'blue'})
for text in texts:
    text.set_color('darkred')
    text.set_fontsize(12)
for pie in pies:
    pie.set_linewidth(0)

plt.legend(title='Country', title_fontsize='10', fontsize='10', loc='lower left', bbox_to_anchor=(0.87, 0.01))
plt.show()


#%%

# Histplot with KDE
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, color='#97d6b9', alpha=0.6, bins=40, line_kws={'linewidth': 1})
plt.title('Distribution of Prices for Cars', fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'blue'})
plt.xlabel('Price', fontdict={'fontsize': 14, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Frequency', fontdict={'fontsize': 14, 'fontname': 'serif', 'color': 'darkred'})
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}.00 k'.format(x/1000)))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}.00 k'.format(x/1000)))
plt.show()

#%%

#QQ-plot

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111)
res = stats.probplot(data['mpg'], dist="norm", plot=ax, rvalue=True)
ax.get_lines()[0].set_markerfacecolor('none')  # Makes the markers hollow
ax.get_lines()[0].set_markeredgecolor('#97d6b9')  # Sets the color of the ring
ax.get_lines()[0].set_marker('o')  # Sets the marker to a circle
ax.get_lines()[1].set_color('#1f80b8')
ax.get_lines()[1].set_linestyle('-')
ax.set_title('Normal QQ-Plot of Car Miles Per Gallon.',fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'blue'})
ax.set_xlabel('Theoretical Quantiles', fontdict={'fontsize': 14, 'fontname': 'serif', 'color': 'darkred'})
ax.set_ylabel('Quantiles of MPG', fontdict={'fontsize': 14, 'fontname': 'serif', 'color': 'darkred'})
plt.show()


#%%

avg_tax = data.groupby(['Manufacturer', 'Category'])['tax'].mean().reset_index(name='Average Tax')


#%%

# Area Plot:

num_shades = 12
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()

manufacturers = avg_tax['Manufacturer'].unique()
fig, axs = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, ax in enumerate(axs.flat):
    if i < len(manufacturers):
        manufacturer = manufacturers[i]
        manufacturer_data = avg_tax[avg_tax['Manufacturer'] == manufacturer]
        single_color_palette = [colors[i + 3]]
        manufacturer_data = manufacturer_data.pivot(index='Category', columns='Manufacturer', values='Average Tax')
        manufacturer_data.plot(kind='area', stacked=False, ax=ax,
                               color=single_color_palette, legend=False)
        if i // 3 == 2:
            ax.set_xlabel('Category',fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'darkred'})
        else:
            ax.set_xlabel('')
        if i % 3 == 0:
            ax.set_ylabel('Average Tax',fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'darkred'})
        else:
            ax.set_ylabel('')

        ax.set_title(manufacturer.capitalize(), fontdict={'fontsize': 18, 'fontname': 'serif'})
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
    else:
        break

for j in range(i + 1, 9):
    fig.delaxes(axs.flat[j])
fig.suptitle('Comparative Analysis of Average Car Tax by Category Across Manufacturers', fontsize=24, fontname='serif',
             color='blue',y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.993])
plt.show()


#%%
num_shades = 27
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()

#%%
# Box plot
plt.figure(figsize=(8,5))
ax = sns.boxplot(x='Manufacturer', y='mileage', data=data, palette=colors[5:], fliersize=4, showfliers=True, width=0.7, linewidth=1)
ax.set_title('Mileage Distribution by Manufacturer', fontsize=22, fontname='serif', color='blue')
ax.set_xlabel('Manufacturer', fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'darkred'})
ax.set_ylabel('Mileage', fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'darkred'})
ax.set_xticklabels([tick.get_text().capitalize() for tick in ax.get_xticklabels()])

plt.tight_layout()
plt.show()


#%%
#boxen plot
num_shades = 27
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()
plt.figure(figsize=(8,5))
ax = sns.boxenplot(x='fuelType', y='price', data=data, palette=colors[5:])
ax.set_title('Price Distribution by Fuel Type', fontsize=22, fontname='serif', color='blue')
ax.set_xlabel('Fuel Type', fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'darkred'})
ax.set_ylabel('Price', fontdict={'fontsize': 17, 'fontname': 'serif', 'color': 'darkred'})
ax.set_xticklabels([tick.get_text().capitalize() for tick in ax.get_xticklabels()])

plt.tight_layout()
plt.show()


#%%












#%%

#%%
def thousands_formatter(x, pos):
    return f'{int(x/1000)}k' if x >= 1000 else str(int(x))

#%%

manufacturer_countries = list(data['Manufacturer_Country'].unique())
import matplotlib.ticker as mticker

def slice_by_manufacturer_country(df, country):
    return df[df["Manufacturer_Country"] == country]




num_shades = 16
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[4:]

fig, axs = plt.subplots(2, 3, figsize=(19, 13))

for idx, country in enumerate(manufacturer_countries):
    ax = axs[idx // 3, idx % 3]
    country_data = slice_by_manufacturer_country(data, country)
    sns.regplot(x='mileage', y='price', data=country_data, ax=ax,
                scatter_kws={'s': 30, 'alpha': 0.6, 'edgecolor': 'white', 'color': colors[idx]},
                line_kws={'color': colors[idx]})
    ax.set_ylim(bottom=0)
    ax.set_title(f'{country}', fontsize=15, fontname='serif', )

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontname('serif')
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontname('serif')

    if idx // 3 == 1:
        ax.set_xlabel('Mileage', fontdict={'fontsize': 16, 'fontname': 'serif', 'color': 'darkred'})
    else:
        ax.set_xlabel('')

    if idx % 3 == 0:
        ax.set_ylabel('Price', fontdict={'fontsize': 16, 'fontname': 'serif', 'color': 'darkred'})
    else:
        ax.set_ylabel('')

fig.suptitle('Comparative Price vs. Mileage across Different Countries', fontsize=27, fontname='serif', color='blue')
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()
#%%




#######


#%%


#%%



colors= ['#c4e8b4', '#97d6b9', '#6dc6be', '#5db5ad']
top_years = data['year'].value_counts().nlargest(10).index
top_years = data['year'].value_counts().nlargest(10).index.sort_values()
df_top_years = data[data['year'].isin(top_years)]

plt.figure(figsize=(13,6))

ax= sns.lineplot(data=df_top_years, x='year', y='mileage', hue='transmission', estimator='mean', ci=None, marker='o',markersize=8, markeredgecolor='grey',palette=colors,linewidth=2)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
for label in ax.get_xticklabels():
    label.set_fontsize(12)
    label.set_fontname('serif')

for label in ax.get_yticklabels():
    label.set_fontsize(12)
    label.set_fontname('serif')
plt.xticks(top_years)
plt.xlabel('Year',fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Average Mileage',fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.title('Average Mileage Across Top 10 Years by Sales Volume', fontsize=21, fontname='serif', color='blue')
plt.legend(title='Transmission Type')
plt.grid(True)
plt.tight_layout()
plt.show()



#%%

import random

num_shades = 50
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()
selected_colors = random.sample(colors, 4)

print(selected_colors)

#%%





#%%

# Stack barplot

num_shades = 10
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]

fig, ax = plt.subplots(figsize=(10, 6))  # Reduced width from 13 to 10
average_price_by_country_fuel = data.groupby(['Manufacturer_Country', 'fuelType'])['price'].mean().unstack(fill_value=0)
average_price_by_country_fuel.plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.65)
ax.set_title('Average Price by Manufacturer Country and Fuel Type', fontsize=20, fontname='serif', color='blue')
ax.set_xlabel('Manufacturer Country', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
ax.set_ylabel('Average Price', fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
ax.set_xticklabels(ax.get_xticklabels(), rotation=360)
ax.legend(title='Fuel Type')
for label in ax.get_xticklabels():
    label.set_fontsize(10)
    label.set_fontname('serif')
for label in ax.get_yticklabels():
    label.set_fontsize(10)
    label.set_fontname('serif')
plt.tight_layout()
plt.show()






#%%

grouped_data = data.groupby(['Manufacturer_Country', 'transmission'])['price'].mean().reset_index()




#%%


# Group barplot

plt.figure(figsize=(12, 7))
num_shades = 12
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]

sns.barplot(x='Manufacturer_Country', y='price', hue='transmission', data=grouped_data, palette=colors)
plt.title('Average Car Price by Transmission Type Across Manufacturer Countries',fontsize=20, fontname='serif', color='blue')
plt.xlabel('Manufacturer Country',fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Average Price',fontdict={'fontsize': 18, 'fontname': 'serif', 'color': 'darkred'})
plt.legend(title='Transmission Type', loc='upper right', title_fontsize='13', fontsize='10')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


#%%

#%%

# Violin plot draft 01:

plt.figure(figsize=(9,5))
num_shades = 10
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]
sns.violinplot(x='fuelType', y='price', data=data,linewidth=0.85,palette=colors,edgecolor='grey')
plt.title('Mileage Distribution by Fuel Type',fontsize=15, fontname='serif', color='blue')
plt.xlabel('Fuel Type',fontdict={'fontsize': 12, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Price',fontdict={'fontsize': 12, 'fontname': 'serif', 'color': 'darkred'})
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

#%%

data.head()
#%%

# Joint Plot with KDE:

plt.figure(figsize=(8,9))

num_shades = 12
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]

joint_kde_scatter = sns.jointplot(x='price', y='tax', data=data, kind='kde', fill=True,color=colors[3],space=0, cmap='Reds')
joint_kde_scatter.plot_joint(sns.scatterplot, s=30, edgecolor='white', linewidth=1, alpha=0.6,color=colors[5])

joint_kde_scatter.set_axis_labels('Price', 'Tax', fontdict={'fontsize': 16, 'fontname': 'serif', 'color': 'darkred'})

joint_kde_scatter.fig.suptitle('Price vs. Tax: KDE and Scatter Representation', fontsize=20, fontname='serif', color='blue')

joint_kde_scatter.fig.subplots_adjust(top=0.92)

joint_kde_scatter.fig.set_size_inches(10, 8)

plt.show()



#%%

sns.set(style="whitegrid")

#%%

# KDE Plot

num_shades = 10
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]

plt.figure(figsize=(10, 6))
sns.kdeplot(data=data, x='price', hue='Category', fill=True, common_norm=False, palette=colors, alpha=0.6, linewidth =1)
plt.title('Price Distribution by Category', fontsize=16, fontname='serif', color='blue')
plt.xlabel('Price', fontdict={'fontsize': 12, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Density', fontdict={'fontsize': 12, 'fontname': 'serif', 'color': 'darkred'})
plt.grid(True,alpha=0.6)
plt.show()

#%%

sns.set(style="darkgrid")
# Cluster map:

numeric_data = data.select_dtypes(include=['number'])

corr = numeric_data.corr()
cluster_map = sns.clustermap(corr, cmap="YlGnBu", figsize=(16, 9), linewidths=.5, annot=True, fmt=".2f",
                             annot_kws={'size': 12, 'fontname': 'serif'},
                             dendrogram_ratio=(.1, .2))

cluster_map.fig.suptitle('Correlation Matrix Cluster Map', fontsize=19, fontname='serif', color='blue', va='center',y=0.95)

cluster_map.ax_heatmap.set_xlabel('Variables', fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})
cluster_map.ax_heatmap.set_ylabel('Variables', fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})

plt.setp(cluster_map.ax_heatmap.get_xticklabels(), fontsize=12, fontname='serif')
plt.setp(cluster_map.ax_heatmap.get_yticklabels(), fontsize=12, fontname='serif')

plt.tight_layout()
plt.show()


#%%

# Hexbin:


sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
edge_color_with_alpha = (0, 0, 0, 0.3)  # Black with 50% transparency
plt.hexbin(data['engineSize'], data['tax'], gridsize=20, cmap='YlGnBu', mincnt=1,
           edgecolors=edge_color_with_alpha)
plt.colorbar(label='Number of cars')
plt.xlabel('Engine Size (L)',fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Tax ($)',fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})
plt.grid(True)
plt.title('Engine Size vs Tax',fontsize=16, fontname='serif', color='blue', va='center',y =1.02)
plt.show()


#%%

# Strip plot

num_shades = 12
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[:]

total_tax_by_year = data.groupby('year')['tax'].sum().reset_index()
#%%
#%%
top_years_by_tax = total_tax_by_year.sort_values('tax', ascending=False).head(6)['year']

filtered_data = data[data['year'].isin(top_years_by_tax)]

manufacturer_order = ['Ford', 'Vauxhall', 'Skoda', 'Hyundi', 'Toyota', 'Mercedes', 'Bmw', 'Volkswagen', 'Audi']
#%%

num_shades = 7
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[1:]



#%%


fig, axs = plt.subplots(2, 3, figsize=(20, 12))
axs = axs.flatten()

for i, year in enumerate(sorted(top_years_by_tax)):
    year_data = filtered_data[filtered_data['year'] == year]

    sns.stripplot(x='Manufacturer', y='tax', data=year_data ,jitter=True, ax=axs[i],color='green',alpha=0.6)
    axs[i].set_title(f' {year}',fontsize=16, fontname='serif', color='navy')
    axs[i].set_xticklabels([text.capitalize() for text in manufacturer_order])

    if i in [3, 4,5]:
        axs[i].set_xlabel('Manufacturer',fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})
    else:
        axs[i].set_xlabel('')

    if i in [0, 3]:
        axs[i].set_ylabel('Tax ($)',fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})
    else:
        axs[i].set_ylabel('')

fig.suptitle('Tax Distribution of Top 6 Years with Highest Car Taxes', fontsize=22, fontname='serif', color='blue')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.tight_layout()
plt.show()



#%%

# Swarm plot:
num_shades = 10
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]
top_manufacturers = data['Manufacturer'].value_counts().index.tolist()

# Filter data for top manufacturers
top_manufacturers_data = data[data['Manufacturer'].isin(top_manufacturers)]


sampled_data = top_manufacturers_data.sample(n=500, random_state=42)

# Create the swarm plot
plt.figure(figsize=(10, 6))
swarm_plot = sns.swarmplot(
    x='fuelType',
    y='mpg',
    data=sampled_data,
    order=sampled_data['fuelType'].value_counts().index,
    palette=colors)



# Set title and labels with formatting
swarm_plot.set_title('Mileage Distribution by Fuel of 500 Samples', fontsize=22, fontname='serif', color='blue')
swarm_plot.set_xlabel('Fuel Type', fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})
swarm_plot.set_ylabel('Mileage per Gallon', fontdict={'fontsize': 15, 'fontname': 'serif', 'color': 'darkred'})

plt.show()

#%%

# Rugplot

num_shades = 10
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]


plt.figure(figsize=(10, 3))
sns.rugplot(data=data, x='mpg',height=0.5, color=colors[2],alpha=0.6)
plt.title('Mileage Per Gallon Distribution of Vehicles',fontsize=14, fontname='serif', color='blue')
plt.xlabel('Mileage Per Gallon',fontdict={'fontsize': 11, 'fontname': 'serif', 'color': 'darkred'})
plt.ylabel('Density',fontdict={'fontsize': 11, 'fontname': 'serif', 'color': 'darkred'})
plt.tight_layout()
plt.show()


#%%


# 3D plot
#%%
#%%

x = np.linspace(data['price'].min(), data['price'].max(), 100)  # Assuming 'price' ranges from 1 to 10 for demonstration
y = np.linspace(data['engineSize'].min(), data['engineSize'].max(), 100)   # Assuming 'engineSize' ranges from 1 to 3 for demonstration
X, Y = np.meshgrid(x, y)
def some_function(X, Y):
    return np.sin(X) + np.cos(Y)

Z = some_function(X, Y)


fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='YlGnBu_r', edgecolor='none', alpha=0.7)

k=ax.scatter(X, Y, Z, c=Z, cmap='YlGnBu_r', edgecolor='none', alpha=0.7)

ax.set_xlabel('Price (X)',fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})
ax.set_ylabel('Engine Size (Y) Transformed',fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})
ax.set_zlabel('Value Ratio (Z)',fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})
ax.set_title('Price vs. Transformed Engine Size',fontsize=16, fontname='serif', color='blue')
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
fig.colorbar(surf, shrink=0.7, aspect=7, label='Value Ratio')
plt.tight_layout()
plt.show()


#%%



#%%

# Distplot





fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)


transmission_types = ['Manual', 'Semi-Auto', 'Automatic', 'Other']
num_shades = 10
palette = sns.color_palette("YlGnBu", num_shades)
colors = palette.as_hex()[2:]
i=0
for idx, (ax, transmission) in enumerate(zip(axs.flat, transmission_types)):
    subset = data[data['transmission'] == transmission]
    sns.distplot(subset['mileage'],  bins=30, ax=ax, color=colors[i])
    ax.set_title(f'Transmission: {transmission}', fontsize=13.5, fontname='serif', color='navy')
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    if idx < 2:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Mileage', fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})

    if idx % 2 == 0:
        ax.set_ylabel('Density', fontdict={'fontsize': 13, 'fontname': 'serif', 'color': 'darkred'})
    else:
        ax.set_ylabel('')

    i += 1
fig.suptitle('Mileage Distribution by Transmission Type', fontsize=20, fontname='serif', color='blue')
plt.tight_layout()
plt.show()

#%%
# Outlier Detection and removal:
def remove_outliers_iqr(data):
    for column in data.select_dtypes(include=['number']).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        print(f"Column: {column}")
        print(f"Values below {abs(lower_bound):.2f} and above {upper_bound:.2f} are considered outliers for this column.\n")

        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    return data
data=remove_outliers_iqr(data)

print(data.info())


############
# PAIRPLOT #
############

numeric_columns = data.select_dtypes(include=['number'])
#%%
plt.figure(figsize=(15, 10))
pair_plot = sns.pairplot(numeric_columns)
plt.subplots_adjust(top=0.95)
plt.suptitle('Pair Plot of Numeric Columns', fontsize=25, fontname='serif', color='blue', y=0.98)
plt.show()


#%%


#  Contour plot:
from scipy.stats import gaussian_kde

x = data['price']
y = data['mileage']


x_range = np.linspace(x.min(), x.max(), 100)
y_range = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(x_range, y_range)


kde = gaussian_kde([x, y])
Z = kde(np.vstack([X.ravel(), Y.ravel()]))


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(X, Y, Z.reshape(X.shape), 50, cmap='YlGnBu_r')
ax.set_xlabel('Price', fontsize=18, fontname='serif', color='darkred')
ax.set_ylabel('Mileage', fontsize=18, fontname='serif', color='darkred')
ax.set_zlabel('Density', fontsize=18, fontname='serif', color='darkred')
ax.set_title('3D Contour Plot of Price vs Mileage', fontsize=25, fontname='serif', color='blue')
plt.show()