import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
import json
from folium import Marker
from folium.plugins import MarkerCluster
from jinja2 import Template
import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests
import scrapy

with st.echo(code_location='below'):
    ### FROM: https://discuss.streamlit.io/t/horizontal-radio-buttons/2114/8
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
             unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
             unsafe_allow_html=True)
    ### END FROM

    st.write("""Implemented functions:\n
       Advanced Pandas (merge, map, MultiIndex);\n
       Advanced Web-scrapping (Scrapy);\n
       Visualization (folium is not trivial, as it show means on clusters);\n
       Streamlit;\n
       GEO (folium);\n
       Machine Learning (sklearn);\n
       Networkx""")

    st.write("Main topic: suicide rates and risk factors in different countries")


    class BlogSpider(scrapy.Spider):
        name = 'wikispider'
        start_urls = ['https://en.wikipedia.org/wiki/World_Happiness_Report']

        def parse(self, response):
            table_xpath = './/h3/span[@id="2019_report"]/../following-sibling::div/table//table'
            table = response.xpath(table_xpath + '/tbody')[0]
            keys = ['rank', '', 'country', 'happiness_score', 'gdp_per_capita', 'soc_support', 'life_exp',
                    'choice_freedom', 'generosity', 'corruption']
            for i, tr in enumerate(table.xpath('./tr')):
                row = tr.xpath('./td//text()').getall()
                vals = row if row else [float('nan')] * len(keys)
                vals = [str(val).strip() for val in vals]
                d = dict(zip(keys, vals))
                d['year'] = 2019
                yield d


    def locations_by_country():
        url = 'https://gist.githubusercontent.com/tadast/8827699/raw/f5cac3d42d16b78348610fc4ec301e9234f82821/countries_codes_and_coordinates.csv'
        df = (pd.read_csv(url).iloc[:, [0, 4, 5]]
              .rename(columns={'Latitude (average)': 'lat', 'Longitude (average)': 'lon',
                               'Country': 'country'})
              .applymap(lambda x: x.strip(' "'))
              .set_index('country')
              .astype('float')
              .assign(coordinates=lambda x: list(zip(x.lat, x.lon)))
              .coordinates
              )
        return df.to_dict()


    country_to_location = locations_by_country()


    def get_attrs_df():
        df = pd.read_csv('18ec75a3-bbac-4251-b0d0-d2e8380c43e3_Data.csv')
        df.drop(columns=['Series Code', 'Country Code',

                         ], inplace=True)
        df.rename(columns={'Series Name': 'measure',
                           'Country Name': 'country',
                           '2016 [YR2016]': '2016',
                           '2017 [YR2017]': '2017',
                           '2018 [YR2018]': '2018',
                           '2019 [YR2019]': '2019'},
                  inplace=True)
        df = df.replace({'..': float('nan')}).dropna()
        return df


    attrs_df = get_attrs_df()


    def get_attrs_year(df=attrs_df):
        pivot = df.pivot(values='2016', index='country', columns='measure').assign(year='2016')
        for year in ['2017', '2018', '2019']:
            pivot = pd.concat((pivot, df.pivot(values=year, index='country', columns='measure').assign(year=year)))
        pivot.drop(columns=
        [
            'Unemployment with advanced education (% of total labor force with advanced education)',
            'Unemployment with advanced education, male (% of male labor force with advanced education)',
            'Unemployment with basic education, female (% of female labor force with basic education)',
            'Unemployment, female (% of female labor force) (national estimate)',
            'Unemployment, male (% of male labor force) (national estimate)',
            'Unemployment, total (% of total labor force) (national estimate)',
            'Research and development expenditure (% of GDP)',
            'Female share of employment in senior and middle management (%)',
            'Poverty gap at $5.50 a day (2011 PPP) (%)',
            'Multidimensional poverty headcount ratio (% of total population)',
            'Multidimensional poverty headcount ratio, female (% of female population)',
            'Multidimensional poverty headcount ratio, male (% of male population)',
        ]
            , inplace=True)
        return pivot


    pivot_attrs = get_attrs_year()


    def get_sh_df():
        df = pd.read_csv('IHME-GBD_2019_DATA-59069656-1.csv')
        df.drop(columns=['measure_id', 'age_id', 'cause_id', 'cause_name', 'location_id', 'age_name', 'metric_id',
                         'metric_name', 'upper', 'lower', 'sex_id'], inplace=True)
        df.rename(columns={'measure_name': 'measure', 'location_name': 'country', 'sex_name': 'sex'}, inplace=True)
        df = df.assign(location=lambda x: x.country.map(country_to_location)).dropna()
        df.replace({'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, inplace=True)
        df.year = df.year.astype('str')
        cols = list(df.columns)
        cols.remove('val')
        cols.remove('location')
        cols.remove('measure')
        cols.remove('sex')
        df = df.merge(pivot_attrs.reset_index(), how='left', on=['country', 'year'])
        df = df.merge(h_df, how='left', on=['country', 'year'])
        return df.set_index(['measure', 'year', 'country', 'sex'])


    def get_happy_df():
        df = pd.read_csv('table.csv').replace('', float('nan')).iloc[:, 2:].dropna()
        df.year = df.year.astype('int').astype('str')
        return df


    h_df = get_happy_df()
    sh_df = get_sh_df()

    measures_df = sh_df.columns[2:]
    selected_measure_df = st.multiselect('Select measures', measures_df, default='corruption')
    show_df = (pd.concat((sh_df['val'], sh_df[selected_measure_df]), axis=1)
               .sort_values(by=selected_measure_df[-1], ascending=False))
    st.dataframe(show_df)

    st.header("Map")


    ### FROM: https://stackoverflow.com/questions/56842575/how-to-display-averages-instead-of-counts-on-folium-markerclusters
    class MarkerWithProps(Marker):
        _template = Template(u"""
            {% macro script(this, kwargs) %}
            var {{this.get_name()}} = L.marker(
                [{{this.location[0]}}, {{this.location[1]}}],
                {
                    icon: new L.Icon.Default(),
                    {%- if this.draggable %}
                    draggable: true,
                    autoPan: true,
                    {%- endif %}
                    {%- if this.props %}
                    props : {{ this.props }} 
                    {%- endif %}
                    }
                )
                .addTo({{this._parent.get_name()}});
            {% endmacro %}
            """)

        def __init__(self, location, popup=None, tooltip=None, icon=None,
                     draggable=False, props=None):
            super(MarkerWithProps, self).__init__(location=location, popup=popup, tooltip=tooltip, icon=icon,
                                                  draggable=draggable)
            self.props = json.loads(json.dumps(props))


    icon_create_function = '''
        function(cluster) {
            var markers = cluster.getAllChildMarkers();
            var sum = 0;
            for (var i = 0; i < markers.length; i++) {
                sum += markers[i].options.props.metric;
            }
            var avg = sum/cluster.getChildCount();
            avg = avg.toFixed(2);
    
            return L.divIcon({
                 html: '<b>' + avg + '</b>',
                 className: 'marker-cluster marker-cluster-small',
                 iconSize: new L.Point(20, 20)
            });
        }
    '''


    ### END FROM

    def get_iframe(measure='Deaths', year='2016', country='Malta', sex='Both'):
        popup = f"<h4>{measure} per 100 000 in {year}:</h4><br>{sh_df.loc[(measure, year, country, sex), 'val']}"
        return popup


    def get_color(measure='Deaths', year='2016', country='Malta', sex='Both'):
        value = sh_df.loc[(measure, year, country, sex), 'val']
        q = lambda x: sh_df.xs((measure, year, sex), level=(0, 1, 3)).val.quantile(x)
        if value <= q(0.25):
            return 'lightgreen'
        elif q(0.25) < value <= q(0.5):
            return 'green'
        elif q(0.5) < value <= q(0.75):
            return 'orange'
        else:
            return 'red'


    measure = st.radio('measure', ['Deaths', 'DALYs'])
    year = st.selectbox('Year: ', sh_df.index.levels[1].unique())
    sex = st.radio('Sex', ['Male', 'Female', 'Both'])

    world_map = folium.Map(location=[37.7622, -122.4356], zoom_start=2)
    marker_cluster = MarkerCluster(icon_create_function=icon_create_function)
    countries = sh_df.xs((measure, year, sex), level=(0, 1, 3)).index
    for country in countries:
        iframe = folium.IFrame(get_iframe(measure, year, country, sex))
        popup = folium.Popup(iframe,
                             min_width=220,
                             max_width=220)
        value = sh_df.loc[(measure, year, country, sex), 'val']
        location = sh_df.loc[(measure, year, country, sex), 'location']
        MarkerWithProps(location=location, popup=popup, tooltip=country,
                        icon=folium.Icon(color=get_color(measure, year, country, sex)),
                        props={'metric': value},
                        ).add_to(marker_cluster)
    marker_cluster.add_to(world_map)

    folium_static(world_map)

    st.header("Regression")
    measure = st.radio('Select measure for regression', ['Deaths', 'DALYs'])
    sex = st.radio('Select sex for regression', ['Male', 'Female', 'Both'])
    possible_attrs = ['happiness_score', 'gdp_per_capita', 'soc_support',
                      'life_exp', 'choice_freedom', 'generosity', 'corruption']
    selected_attrs = st.multiselect('Select attributes for regression', possible_attrs, default=possible_attrs)
    df = (sh_df.xs((measure, '2019', sex), level=[0, 1, 3])
          .loc[:, ['val'] + selected_attrs].dropna())
    attrs = df.loc[:, selected_attrs]
    main_text = st.selectbox('Select main attribute', selected_attrs) if len(selected_attrs) > 0 else selected_attrs


    def create_regression(attrs, df, main_text):
        goal = df.val
        model = LinearRegression()
        model.fit(attrs, goal)
        y = model.predict(attrs)
        fig, ax = plt.subplots()
        ax.scatter(x=df[main_text], y=goal)
        ax.plot(df[main_text], y, 'o', color='C1', alpha=0.5)
        return fig


    st.pyplot(create_regression(attrs, df, main_text))

    st.header('Grapgh')
    st.subheader('Correlation between different measures')
    st.text(
        """Nodes are selected measure, happiness_score, gdp_per_capita,
        soc_support, life_expectancy, choice_freedom, generosity,
        and corruption perception""")

    measure = st.radio('Select measure for graph', ['Deaths', 'DALYs'])
    threshold = st.slider('Threshold (abs)', 0.0, 1.0, 0.2, 0.005)
    sign = st.radio("Positive or negative correlation", ('+', '-'), 1)


    def draw_graph(measure, threshold, sign):
        fig, ax = plt.subplots()
        df = sh_df.loc[measure].corr()
        names = [name[:3] for name in df.index]
        cond = lambda x: x > threshold if sign == '+' else x < -threshold

        coors = np.array([(x, y) for x in range(3) for y in range(3)])
        pos = dict(zip(names, coors))
        pos['val'], pos['lif'] = pos['lif'], pos['val']
        G = nx.Graph()

        for i, name in enumerate(names):
            row = df.iloc[i, :]
            G.add_node(name)
            for j, name2 in enumerate(names):
                if cond(row.iloc[j]) and j > i:
                    G.add_edge(name, name2)

        options = {
            "font_size": 14,
            "node_size": 2000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
        }
        nx.draw_networkx(G, pos, **options)
        return fig


    st.pyplot(draw_graph(measure, threshold, sign))
    st.write("As you can see, there's a somewhat strong correlation between death rate and life expectancy")

    st.header("(API) Comparison with 2004")
    deaths_code = 'SA_0000001453'
    dalys_code = 'SA_0000001434'


    @st.experimental_memo
    def get_data_from_who(code, metric):
        print('WHO data')
        storage_options = {'User-Agent': 'Mozilla/5.0'}
        df = pd.read_csv(f'https://apps.who.int/gho/athena/api/GHO/{code}?format=csv',
                         storage_options=storage_options)
        df = df[['YEAR', 'REGION', 'COUNTRY', 'SEX', 'Display Value']]
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'display value': metric})
        return df


    @st.experimental_memo
    def get_countries_codes():
        print('getting info from world bank')
        codes_dict = dict()
        url = f'http://api.worldbank.org/v2/country'
        for page in range(1, 7):
            params = {'format': 'json', 'page': page}
            codes_json = requests.get(url, params)
            codes_list = codes_json.json()[1]
            for country in codes_list:
                codes_dict[country['id']] = country['name']
        return codes_dict


    deaths_df_old = get_data_from_who('SA_0000001453', 'Deaths')
    DALYs_df_old = get_data_from_who('SA_0000001434', 'DALYs')
    sh_df_old = deaths_df_old.merge(DALYs_df_old, how='outer', on=['year', 'region', 'country', 'sex'])

    sh_df_old.replace(
        {'country': get_countries_codes(),
         'sex': {'MLE': 'Male', 'FMLE': 'Female', 'BTSX': 'Both'}},
        inplace=True)
    sh_df_old = sh_df_old[sh_df_old.year == 2004].drop(columns=['region', 'year'])
    measure_compare = st.radio('Measure to compare', ['Deaths', 'DALYs'])
    sex_compare = st.radio('Sex to compare', ['Male', 'Female', 'Both'])
    sh_df_old = sh_df_old[sh_df_old.sex == sex_compare].loc[:, ['country', measure_compare]]
    df_compare = sh_df.xs((measure_compare, '2019', sex_compare), level=[0, 1, 3]).reset_index().iloc[:, :2].rename(
        columns={})
    total_compare = (sh_df_old.merge(df_compare, how='inner', on='country')
                     .assign(diff=lambda x: x.val - x.loc[:, measure_compare],
                             diff_percent=lambda x:round(100*(x.loc[:, 'diff']/x.loc[:, measure_compare]),1)))
    st.dataframe(total_compare)
