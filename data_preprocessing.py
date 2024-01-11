
from pandas import read_csv
from pandas import set_option
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

def read_data():
    filename = 'cardio_data_re.csv'         # 加载处理后数据集
    data = read_csv(filename)
    data = data.iloc[:, 1:]
    names = data.columns.values
    print(names)

    print(data.isnull().any())
    # 显示每一种特征元素均无缺失值
    # 数据维度
    print(data.shape)

    # 查看数据类型
    set_option('display.max_rows', 500)
    print(data.dtypes)

    # 查看前20条记录
    set_option('display.max_columns', 1000)
    set_option('display.max_colwidth', None)
    set_option('max_colwidth', 100)
    set_option('display.width', 1000)
    print(data.head(20))

    # 描述性统计量
    set_option('display.precision', 3)
    print(data.describe())

    # 数据的分类分布
    print(data.groupby('cardio').size())
    return data

def split_data(data):
    # 分离评估数据集

    # 选择筛选后的变量
    data = data[['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'smoke', 'alco', 'active', 'cardio']]

    # 箱线图
    data_1 = data[['age', 'weight', 'ap_hi', 'ap_lo']]
    data_1.plot(kind='box', subplots=True, layout=(1, 4), sharex=False, sharey=False, fontsize=8)
    # pyplot.savefig("box_plot.svg", dpi=300, format="svg")
    pyplot.show()

    X = data.drop(columns=['cardio'])
    Y = data.cardio


    validation_size = 0.2
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

    return X, Y, X_train, X_validation, Y_train, Y_validation