这是scikit-learn的类和函数参考。请参阅完整的[用户指南](https://scikit-learn.org.cn/lists/2.html)以获取更多详细信息，因为类和函数的原始规范可能不足以提供有关其用法的完整指南。有关在API上重复的概念的参考，请参阅[“通用术语表和API元素”](https://scikit-learn.org.cn/lists/91.html)。

## sklearn.base：基类和实用程序函数

所有估计量的基类。

用于投票分类器

### 基类

| [`base.BaseEstimator`](https://scikit-learn.org.cn/view/78.html) | scikit-learn中所有估计器的基类 |
| --- | --- |
| [`base.BiclusterMixin`](https://scikit-learn.org.cn/view/79.html) | scikit-learn中所有双簇估计器的Mixin类 |
| [`base.ClassifierMixin`](https://scikit-learn.org.cn/view/82.html) | scikit-learn中所有分类器的Mixin类。 |
| [`base.ClusterMixin`](https://scikit-learn.org.cn/view/356.html) | scikit-learn中所有聚类估计器的Mixin类。 |
| [`base.DensityMixin`](https://scikit-learn.org.cn/view/357.html) | scikit-learn中所有密度估计器的Mixin类。 |
| [`base.RegressorMixin`](https://scikit-learn.org.cn/view/358.html) | scikit-learn中所有回归估计器的Mixin类。 |
| [`base.TransformerMixin`](https://scikit-learn.org.cn/view/359.html) | scikit-learn中所有转换器的Mixin类。 |
| [`feature_selection.SelectorMixin`](https://scikit-learn.org.cn/view/360.html) | 给定支持掩码、可以执行特征选择的转换器的Mixin类。 |

### 函数

| [`base.clone`](https://scikit-learn.org.cn/view/361.html)(estimator, \*\[, safe\]) | 构造一个具有相同参数的新估算器。 |
| --- | --- |
| [`base.is_classifier`](https://scikit-learn.org.cn/view/362.html)(estimator) | 如果给定的估计器（可能）是分类器，则返回True。 |
| [`base.is_regressor`](https://scikit-learn.org.cn/view/363.html)(estimator) | 如果给定的估计器（可能）是回归器，则返回True。 |
| [`config_context`](https://scikit-learn.org.cn/view/364.html)（\*\* new\_config） | 全局scikit-learn配置的上下文管理器 |
| [`get_config`](https://scikit-learn.org.cn/view/365.html)（） | 检索[`set_config`](https://scikit-learn.org.cn/view/366.html)配置的当前值 |
| [`set_config`](https://scikit-learn.org.cn/view/366.html)(\[assume\_finite, working\_memory, …\]) | 设置全局scikit-learn配置 |
| [`show_versions`](https://scikit-learn.org.cn/view/367.html)（） | 打印有用的调试信息 |

## sklearn.calibration：概率校准

校准预测概率。

**用户指南**：有关更多详细信息，请参见“[概率校准](https://scikit-learn.org.cn/view/104.html)”部分。

| [`calibration.CalibratedClassifierCV`](https://scikit-learn.org.cn/view/368.html)（\[…\]） | 等渗回归或逻辑回归的概率校正。 |
| --- | --- |
| More Actions[`calibration.calibration_curve`](https://scikit-learn.org.cn/view/369.html)（y\_true，y\_prob，\*） | 计算校准曲线的真实和预测概率。 |

## sklearn.cluster：聚类

该[`sklearn.cluster`](https://scikit-learn.org.cn/lists/3.html#sklearn.cluster%EF%BC%9A%E8%81%9A%E7%B1%BB)模块收集了流行的无监督聚类算法。

**用户指南**：有关更多详细信息，请参见“ [聚类](https://scikit-learn.org.cn/view/108.html)“和”[双聚类](https://scikit-learn.org.cn/view/109.html)“部分。

### 类

| [`cluster.AffinityPropagation`](https://scikit-learn.org.cn/view/370.html)(\*\[, damping, …\]) | 执行数据的相似性传播聚类。 |
| --- | --- |
| [`cluster.AgglomerativeClustering`](https://scikit-learn.org.cn/view/371.html)（\[…\]） | 聚集聚类 |
| [`cluster.Birch`](https://scikit-learn.org.cn/view/375.html)(\*\[, threshold, …\]) | 实现Birch聚类算法。 |
| [`cluster.DBSCAN`](https://scikit-learn.org.cn/view/379.html)(\[eps, min\_samples, metric, …\]) | 从向量数组或距离矩阵执行DBSCAN聚类。 |
| [`cluster.FeatureAgglomeration`](https://scikit-learn.org.cn/view/380.html)（\[n\_clusters，...\]） | 聚集函数。 |
| [`cluster.KMeans`](https://scikit-learn.org.cn/view/383.html)（\[n\_clusters，init，n\_init，...\]） | K-均值聚类。 |
| [`cluster.MiniBatchKMeans`](https://scikit-learn.org.cn/view/387.html)（\[n\_clusters，init，...\]） | 小批次K均值聚类。 |
| [`cluster.MeanShift`](https://scikit-learn.org.cn/view/389.html)(\*\[, bandwidth, seeds, …\]) | 使用扁平内核的均值漂移聚类。 |
| [`cluster.OPTICS`](https://scikit-learn.org.cn/view/390.html)（\* \[，min\_samples，max\_eps，…\]） | 从向量数组估计聚类结构。 |
| [`cluster.SpectralClustering`](https://scikit-learn.org.cn/view/391.html)（\[n\_clusters，...\]） | 将聚类应用于规范化拉普拉斯算子的投影。 |
| [`cluster.SpectralBiclustering`](https://scikit-learn.org.cn/view/392.html)（\[n\_clusters，...\]） | 频谱双聚类（Kluger，2003）。 |
| [`cluster.SpectralCoclustering`](https://scikit-learn.org.cn/view/393.html)（\[n\_clusters，...\]） | 频谱共聚算法（Dhillon，2001）。 |

### 函数

| [`cluster.affinity_propagation`](https://scikit-learn.org.cn/view/395.html)（S，\* \[，...\]） | 执行数据的相似性传播聚类 |
| --- | --- |
| [`cluster.cluster_optics_dbscan`](https://scikit-learn.org.cn/view/396.html)（\*，…） | 对任意epsilon执行DBSCAN提取。 |
| [`cluster.cluster_optics_xi`](https://scikit-learn.org.cn/view/397.html)(\*, reachability, …) | 根据Xi-steep方法自动提取聚类。 |
| [`cluster.compute_optics_graph`](https://scikit-learn.org.cn/view/402.html)（X， \*， …） | 计算OPTICS可达性图。 |
| [`cluster.dbscan`](https://scikit-learn.org.cn/view/403.html)（X \[，eps，min\_samples，…\]） | 从向量数组或距离矩阵执行DBSCAN聚类。 |
| [`cluster.estimate_bandwidth`](https://scikit-learn.org.cn/view/398.html)(X, \*\[, quantile, …\]) | 估计均值漂移算法要使用的带宽。 |
| [`cluster.k_means`](https://scikit-learn.org.cn/view/405.html)（X，n\_clusters，\* \[，…\]） | K-均值聚类算法。 |
| [`cluster.mean_shift`](https://scikit-learn.org.cn/view/406.html)(X, \*\[, bandwidth, seeds, …\]) | 使用扁平内核执行数据的均值漂移聚类。 |
| [`cluster.spectral_clustering`](https://scikit-learn.org.cn/view/408.html)(affinity, \*\[, …\]) | 将聚类应用于规范化拉普拉斯算子的投影。 |
| [`cluster.ward_tree`](https://scikit-learn.org.cn/view/412.html)(X, \*\[, connectivity, …\]) | 基于特征矩阵的Ward聚类。 |

## sklearn.compose：复合估计器

用于使用Transformer转换器构建复合模型的元估计器

除了当前的内容外，这个模块最终将成为Pipeline和FeatureUnion的翻新版本。

**用户指南**：有关更多详细信息，请参见“ [管道和复合估计器](https://scikit-learn.org.cn/view/118.html)”部分。

| [`compose.ColumnTransformer`](https://scikit-learn.org.cn/view/413.html)(transformers, \*\[, …\]) | 将转换器应用于数组或pandas DataFrame的列。 |
| --- | --- |
| [`compose.TransformedTargetRegressor`](https://scikit-learn.org.cn/view/415.html)（\[…\]） | 元估算器，可对转换后的目标进行回归。 |
| [`compose.make_column_transformer`](https://scikit-learn.org.cn/view/423.html)（...） | 从给定的转换器构造一个列转换器。 |
| [`compose.make_column_selector`](https://scikit-learn.org.cn/view/424.html)(\[pattern, …\]) | 创建可调用对象以选择要与`ColumnTransformer`一起使用的列。 |

## sklearn.covariance：协方差估计器

[`sklearn.covariance`](https://scikit-learn.org.cn/lists/3.html#sklearn.covariance%EF%BC%9A%E5%8D%8F%E6%96%B9%E5%B7%AE%E4%BC%B0%E8%AE%A1%E5%99%A8)模块包括可靠地估计给定一组点的特征的协方差的方法和算法。定义为协方差的逆的精度矩阵也被估计。协方差估计与高斯图形模型理论密切相关。

**用户指南**：有关更多详细信息，请参见“[协方差估计](https://scikit-learn.org.cn/view/111.html)”部分。

| [`covariance.EmpiricalCovariance`](https://scikit-learn.org.cn/view/429.html)（\* \[，…\]） | 最大似然协方差估计器 |
| --- | --- |
| [`covariance.EllipticEnvelope`](https://scikit-learn.org.cn/view/432.html)（\* \[，…\]） | 用于检测高斯分布数据集中异常值的对象 |
| [`covariance.GraphicalLasso`](https://scikit-learn.org.cn/view/434.html)(\[alpha, mode, …\]) | 带有l1惩罚估计器的稀疏逆协方差估计 |
| [`covariance.GraphicalLassoCV`](https://scikit-learn.org.cn/view/435.html)（\* \[，alphas，…\]） | 带有l1惩罚的交叉验证选择的稀疏逆协方差 |
| [`covariance.LedoitWolf`](https://scikit-learn.org.cn/view/438.html)（\* \[，store\_precision，…\]） | LedoitWolf估计器 |
| [`covariance.MinCovDet`](https://scikit-learn.org.cn/view/441.html)（\* \[，store\_precision，…\]） | 最小协方差决定因素（MCD）：协方差的稳健估计器 |
| [`covariance.OAS`](https://scikit-learn.org.cn/view/444.html)（\* \[，store\_precision，…\]） | Oracle近似收缩估计 |
| [`covariance.ShrunkCovariance`](https://scikit-learn.org.cn/view/451.html)（\* \[，…\]） | 收缩协方差估计 |
| [`covariance.empirical_covariance`](https://scikit-learn.org.cn/view/453.html)（X， \*\[， …\]） | 计算最大似然协方差估计器 |
| [`covariance.graphical_lasso`](https://scikit-learn.org.cn/view/454.html)（emp\_cov，alpha，\*） | l1惩罚协方差估计器 |
| [`covariance.ledoit_wolf`](https://scikit-learn.org.cn/view/456.html)（X， \*\[， …\]） | 估计收缩的Ledoit-Wolf协方差矩阵 |
| [`covariance.oas`](https://scikit-learn.org.cn/view/457.html)(X, \*\[, assume\_centered\]) | 使用Oracle近似收缩算法估算协方差 |
| [`covariance.shrunk_covariance`](https://scikit-learn.org.cn/view/458.html)（emp\_cov \[，…\]） | 计算对角线上收缩的协方差矩阵 |

## sklearn.cross\_decomposition：交叉分解

**用户指南**：有关更多详细信息，请参见“ [交叉分解](https://scikit-learn.org.cn/view/87.html)”部分。

| [`cross_decomposition.CCA`](https://scikit-learn.org.cn/view/460.html)（\[n\_components，...\]） | CCA典型相关分析。 |
| --- | --- |
| [`cross_decomposition.PLSCanonical`](https://scikit-learn.org.cn/view/462.html)（\[…\]） | PLSCanonical实现了原始Wold算法的2块规范PLS \[Tenenhaus 1998\] p.204，在\[Wegelin 2000\]中称为PLS-C2A。 |
| [`cross_decomposition.PLSRegression`](https://scikit-learn.org.cn/view/466.html)（\[…\]） | PLS回归 |
| [`cross_decomposition.PLSSVD`](https://scikit-learn.org.cn/view/469.html)（\[n\_components，...\]） | 偏最小二乘SVD |

## sklearn.datasets：数据集

该[`sklearn.datasets`](https://scikit-learn.org.cn/lists/3.html#sklearn.datasets%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%9B%86)模块包括用于加载数据集的实用程序，包括用于加载和获取流行的参考数据集的方法。它还具有一些人工数据生成器。

**用户指南**：有关更多详细信息，请参见“ [数据集加载实用程序](https://scikit-learn.org.cn/view/121.html)”部分。

### 加载器

| [`datasets.clear_data_home`](https://scikit-learn.org.cn/view/473.html)（\[data\_home\]） | 删除数据主目录缓存的所有内容。 |
| --- | --- |
| [`datasets.dump_svmlight_file`](https://scikit-learn.org.cn/view/470.html)（X，y，f，\* \[，…\]） | 以svmlight / libsvm文件格式转储数据集。 |
| [`datasets.fetch_20newsgroups`](https://scikit-learn.org.cn/view/475.html)（\* \[，data\_home，…\]） | 从20个新闻组数据集中加载文件名和数据（分类）。 |
| [`datasets.fetch_20newsgroups_vectorized`](https://scikit-learn.org.cn/view/477.html)（\* \[，…\]） | 加载20个新闻组数据集并将其向量化为令牌计数（分类）。 |
| [`datasets.fetch_california_housing`](https://scikit-learn.org.cn/view/483.html)（\* \[，…\]） | 加载加利福尼亚住房数据集（回归）。 |
| [`datasets.fetch_covtype`](https://scikit-learn.org.cn/view/503.html)（\* \[，data\_home，…\]） | 加载covertype数据集（分类）。 |
| [`datasets.fetch_kddcup99`](https://scikit-learn.org.cn/view/504.html)(\*\[, subset, …\]) | 加载kddcup99数据集（分类）。 |
| [`datasets.fetch_lfw_pairs`](https://scikit-learn.org.cn/view/505.html)(\*\[, subset, …\]) | 加载标记过的人脸Wild (LFW) pairs数据集（分类）。 |
| [`datasets.fetch_lfw_people`](https://scikit-learn.org.cn/view/506.html)（\* \[，data\_home，…\]） | 加载标记过的人脸Wild (LFW) people数据集（分类）。 |
| [`datasets.fetch_olivetti_faces`](https://scikit-learn.org.cn/view/507.html)（\* \[，…\]） | 从AT＆T（分类）中加载Olivetti人脸数据集。 |
| [`datasets.fetch_openml`](https://scikit-learn.org.cn/view/508.html)(\[name, version, …\]) | 通过名称或数据集ID从openml获取数据集。 |
| [`datasets.fetch_rcv1`](https://scikit-learn.org.cn/view/878.html)(\*\[, data\_home, subset, …\]) | 加载RCV1多标签数据集（分类）。 |
| [`datasets.fetch_species_distributions`](https://scikit-learn.org.cn/view/512.html)（\* \[，…\]） | Phillips等人的物种分布数据集加载程序。 |
| [`datasets.get_data_home`](https://scikit-learn.org.cn/view/514.html)（\[data\_home\]） | 返回scikit-learn数据目录的路径。 |
| [`datasets.load_boston`](https://scikit-learn.org.cn/view/515.html)（\* \[，return\_X\_y\]） | 加载并返回波士顿房价数据集（回归）。 |
| [`datasets.load_breast_cancer`](https://scikit-learn.org.cn/view/518.html)（\* \[，return\_X\_y，…\]） | 加载并返回威斯康星州乳腺癌数据集（分类）。 |
| [`datasets.load_diabetes`](https://scikit-learn.org.cn/view/520.html)（\* \[，return\_X\_y，as\_frame\]） | 加载并返回糖尿病数据集（回归）。 |
| [`datasets.load_digits`](https://scikit-learn.org.cn/view/532.html)（\* \[，n\_class，…\]） | 加载并返回数字数据集（分类）。 |
| [`datasets.load_files`](https://scikit-learn.org.cn/view/540.html)（container\_path，\* \[，...\]） | 加载带有类别作为子文件夹名称的文本文件。 |
| [`datasets.load_iris`](https://scikit-learn.org.cn/view/542.html)（\* \[，return\_X\_y，as\_frame\]） | 加载并返回鸢尾花数据集（分类）。 |
| [`datasets.load_linnerud`](https://scikit-learn.org.cn/view/543.html)（\* \[，return\_X\_y，as\_frame\]） | 加载并返回linnerud物理锻炼数据集。 |
| [`datasets.load_sample_image`](https://scikit-learn.org.cn/view/545.html)(image\_name) | 加载单个样本图像的numpy数组 |
| [`datasets.load_sample_images`](https://scikit-learn.org.cn/view/546.html)（） | 加载样本图像以进行图像处理。 |
| [`datasets.load_svmlight_file`](https://scikit-learn.org.cn/view/547.html)（F， \*\[， …\]） | 将svmlight / libsvm格式的数据集加载到稀疏CSR矩阵中 |
| [`datasets.load_svmlight_files`](https://scikit-learn.org.cn/view/548.html)(files, \*\[, …\]) | 从SVMlight格式的多个文件加载数据集 |
| [`datasets.load_wine`](https://scikit-learn.org.cn/view/549.html)（\* \[，return\_X\_y，as\_frame\]） | 加载并返回葡萄酒数据集（分类）。 |

### 样本生成器

| [`datasets.make_biclusters`](https://scikit-learn.org.cn/view/552.html)(shape, n\_clusters, \*) | 生成具有恒定块对角线结构的数组以进行双聚类。 |
| --- | --- |
| [`datasets.make_blobs`](https://scikit-learn.org.cn/view/556.html)(\[n\_samples, n\_features, …\]) | 生成各向同性的高斯团簇。 |
| [`datasets.make_checkerboard`](https://scikit-learn.org.cn/view/571.html)(shape, n\_clusters, \*) | 生成具有棋盘格结构的数组以进行二聚类。 |
| [`datasets.make_circles`](https://scikit-learn.org.cn/view/575.html)(\[n\_samples, shuffle, …\]) | 在2维中制作一个包含较小圆圈的大圆圈。 |
| [`datasets.make_classification`](https://scikit-learn.org.cn/view/583.html)（\[n\_samples，...\]） | 生成随机的n类分类问题。 |
| [`datasets.make_friedman1`](https://scikit-learn.org.cn/view/586.html)（\[n\_samples，...\]） | 生成“ Friedman＃1”回归问题 |
| [`datasets.make_friedman2`](https://scikit-learn.org.cn/view/587.html)(\[n\_samples, noise, …\]) | 生成“ Friedman＃2”回归问题 |
| [`datasets.make_friedman3`](https://scikit-learn.org.cn/view/588.html)(\[n\_samples, noise, …\]) | 生成“ Friedman＃3”回归问题 |
| [`datasets.make_gaussian_quantiles`](https://scikit-learn.org.cn/view/589.html)(\*\[, mean, …\]) | 生成各向同性高斯分布，用分位数标注样本 |
| [`datasets.make_hastie_10_2`](https://scikit-learn.org.cn/view/590.html)（\[n\_samples，...\]） | 生成Hastie等人使用的二进制分类数据。 |
| [`datasets.make_low_rank_matrix`](https://scikit-learn.org.cn/view/592.html)（\[n\_samples，...\]） | 生成具有钟形奇异值的低阶矩阵 |
| [`datasets.make_moons`](https://scikit-learn.org.cn/view/593.html)(\[n\_samples, shuffle, …\]) | 做两个交错的半圈 |
| [`datasets.make_multilabel_classification`](https://scikit-learn.org.cn/view/594.html)（\[…\]） | 生成随机的多标签分类问题。 |
| [`datasets.make_regression`](https://scikit-learn.org.cn/view/595.html)（\[n\_samples，...\]） | 产生随机回归问题。 |
| [`datasets.make_s_curve`](https://scikit-learn.org.cn/view/596.html)(\[n\_samples, noise, …\]) | 生成S曲线数据集。 |
| [`datasets.make_sparse_coded_signal`](https://scikit-learn.org.cn/view/597.html)(n\_samples, …) | 生成信号作为字典元素的稀疏组合。 |
| [`datasets.make_sparse_spd_matrix`](https://scikit-learn.org.cn/view/598.html)(\[dim, …\]) | 生成稀疏对称正定矩阵。 |
| [`datasets.make_sparse_uncorrelated`](https://scikit-learn.org.cn/view/599.html)（\[…\]） | 使用稀疏的不相关设计生成随机回归问题 |
| [`datasets.make_spd_matrix`](https://scikit-learn.org.cn/view/600.html)（n\_dim，\* \[，...\]） | 生成随机对称的正定矩阵。 |
| [`datasets.make_swiss_roll`](https://scikit-learn.org.cn/view/601.html)(\[n\_samples, noise, …\]) | 生成瑞士卷数据集。 |

## sklearn.decomposition：矩阵分解

该[`sklearn.decomposition`](https://scikit-learn.org.cn/lists/3.html#sklearn.decomposition%EF%BC%9A%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3)模块包括矩阵分解算法，其中包括PCA，NMF或ICA。该模块的大多数算法都可以视为降维技术。

**用户指南**：有关更多详细信息，请参见"[分解组件中的信号（矩阵分解问题）](https://scikit-learn.org.cn/view/110.html)"部分。

| [`decomposition.DictionaryLearning`](https://scikit-learn.org.cn/view/602.html)（\[…\]） | 字典学习 |
| --- | --- |
| [`decomposition.FactorAnalysis`](https://scikit-learn.org.cn/view/603.html)（\[n\_components，...\]） | 因子分析（FA） |
| [`decomposition.FastICA`](https://scikit-learn.org.cn/view/604.html)（\[n\_components，...\]） | FastICA：一种用于独立成分分析的快速算法。 |
| [`decomposition.IncrementalPCA`](https://scikit-learn.org.cn/view/605.html)（\[n\_components，...\]） | 增量主成分分析（IPCA）。 |
| [`decomposition.KernelPCA`](https://scikit-learn.org.cn/view/879.html)（\[n\_components，...\]） | 内核主成分分析（KPCA） |
| [`decomposition.LatentDirichletAllocation`](https://scikit-learn.org.cn/view/606.html)（\[…\]） | 在线变分贝叶斯算法的潜在狄利克雷分配 |
| [`decomposition.MiniBatchDictionaryLearning`](https://scikit-learn.org.cn/view/607.html)（\[…\]） | 小批量字典学习 |
| [`decomposition.MiniBatchSparsePCA`](https://scikit-learn.org.cn/view/608.html)（\[…\]） | 小批量稀疏主成分分析 |
| [`decomposition.NMF`](https://scikit-learn.org.cn/view/609.html)（\[n\_components，init，...\]） | 非负矩阵分解（NMF） |
| [`decomposition.PCA`](https://scikit-learn.org.cn/view/610.html)(\[n\_components, copy, …\]) | 主成分分析（PCA）。 |
| [`decomposition.SparsePCA`](https://scikit-learn.org.cn/view/880.html)（\[n\_components，...\]） | 稀疏主成分分析（SparsePCA） |
| [`decomposition.SparseCoder`](https://scikit-learn.org.cn/view/611.html)(dictionary, \*\[, …\]) | 稀疏编码 |
| [`decomposition.TruncatedSVD`](https://scikit-learn.org.cn/view/612.html)（\[n\_components，...\]） | 使用截断的SVD（aka LSA）进行降维。 |
| [`decomposition.dict_learning`](https://scikit-learn.org.cn/view/613.html)（X，n\_components，…） | 解决字典学习矩阵分解问题。 |
| [`decomposition.dict_learning_online`](https://scikit-learn.org.cn/view/614.html)（X\[， …\]） | 在线解决字典学习矩阵分解问题。 |
| [`decomposition.fastica`](https://scikit-learn.org.cn/view/615.html)（X \[，n\_components，…\]） | 执行快速独立成分分析。 |
| [`decomposition.non_negative_factorization`](https://scikit-learn.org.cn/view/616.html)（X） | 计算非负矩阵分解（NMF） |
| [`decomposition.sparse_encode`](https://scikit-learn.org.cn/view/617.html)(X, dictionary, \*) | 稀疏编码 |

## sklearn.discriminant\_analysis：判别分析

线性判别分析和二次判别分析

**用户指南**：有关更多详细信息，请参见“ [线性和二次判别分析](https://scikit-learn.org.cn/view/77.html)”部分。

| [`discriminant_analysis.LinearDiscriminantAnalysis`](https://scikit-learn.org.cn/view/618.html)（\*） | 线性判别分析 |
| --- | --- |
| [`discriminant_analysis.QuadraticDiscriminantAnalysis`](https://scikit-learn.org.cn/view/619.html)（\*） | 二次判别分析 |

## sklearn.dummy：虚拟估计器

**用户指南**：有关更多详细信息，请参阅[指标和评分：量化预测的质量](https://scikit-learn.org.cn/view/93.html)部分。

| [`dummy.DummyClassifier`](https://scikit-learn.org.cn/view/620.html)(\*\[, strategy, …\]) | DummyClassifier是使用简单规则进行预测的分类器。 |
| --- | --- |
| [`dummy.DummyRegressor`](https://scikit-learn.org.cn/view/621.html)(\*\[, strategy, …\]) | DummyRegressor是使用简单规则进行预测的回归器。 |

## sklearn.ensemble：集成方法

该[`sklearn.ensemble`](https://scikit-learn.org.cn/lists/3.html#sklearn.ensemble%EF%BC%9A%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95)模块包括基于集成的分类，回归和异常检测方法。

**用户指南**：有关更多详细信息，请参见[集成方法](https://scikit-learn.org.cn/view/90.html)部分。

| [`ensemble.AdaBoostClassifier`](https://scikit-learn.org.cn/view/622.html)（\[…\]） | AdaBoost分类器。 |
| --- | --- |
| [`ensemble.AdaBoostRegressor`](https://scikit-learn.org.cn/view/623.html)（\[base\_estimator，...\]） | AdaBoost回归器。 |
| [`ensemble.BaggingClassifier`](https://scikit-learn.org.cn/view/624.html)（\[base\_estimator，...\]） | 装袋分类器。 |
| [`ensemble.BaggingRegressor`](https://scikit-learn.org.cn/view/625.html)（\[base\_estimator，...\]） | 装袋回归器。 |
| [`ensemble.ExtraTreesClassifier`](https://scikit-learn.org.cn/view/626.html)（\[…\]） | 极端树分类器。 |
| [`ensemble.ExtraTreesRegressor`](https://scikit-learn.org.cn/view/627.html)（\[n\_estimators，…\]） | 极端树回归器。 |
| [`ensemble.GradientBoostingClassifier`](https://scikit-learn.org.cn/view/628.html)（\* \[，…\]） | 用于分类的梯度提升。 |
| [`ensemble.GradientBoostingRegressor`](https://scikit-learn.org.cn/view/629.html)（\* \[，…\]） | 用于回归的梯度提升。 |
| [`ensemble.IsolationForest`](https://scikit-learn.org.cn/view/631.html)（\* \[，n\_estimators，…\]） | 孤立森林算法。 |
| [`ensemble.RandomForestClassifier`](https://scikit-learn.org.cn/view/633.html)（\[…\]） | 随机森林分类器。 |
| [`ensemble.RandomForestRegressor`](https://scikit-learn.org.cn/view/650.html)（\[…\]） | 随机森林回归器。 |
| [`ensemble.RandomTreesEmbedding`](https://scikit-learn.org.cn/view/651.html)（\[…\]） | 完全随机树的集合。 |
| [`ensemble.StackingClassifier`](https://scikit-learn.org.cn/view/652.html)(estimators\[, …\]) | 带有最终分类器的估计器堆栈。 |
| [`ensemble.StackingRegressor`](https://scikit-learn.org.cn/view/653.html)(estimators\[, …\]) | 带有最终回归器的估计器堆栈。 |
| [`ensemble.VotingClassifier`](https://scikit-learn.org.cn/view/654.html)(estimators, \*\[, …\]) | 针对不拟合估计器的软投票或多数规则分类器。 |
| [`ensemble.VotingRegressor`](https://scikit-learn.org.cn/view/659.html)(estimators, \*\[, …\]) | 对不拟合估计器的预测投票回归。 |
| [`ensemble.HistGradientBoostingRegressor`](https://scikit-learn.org.cn/view/661.html)（\[…\]） | 基于直方图的梯度提升回归树。 |
| [`ensemble.HistGradientBoostingClassifier`](https://scikit-learn.org.cn/view/664.html)（\[…\]） | 基于直方图的梯度提升分类树。 |

## sklearn.exceptions：异常和警告

该[`sklearn.exceptions`](https://scikit-learn.org.cn/lists/3.html#sklearn.exceptions%EF%BC%9A%E5%BC%82%E5%B8%B8%E5%92%8C%E8%AD%A6%E5%91%8A)模块包括scikit-learn中使用的所有自定义警告和错误类。

| [`exceptions.ChangedBehaviorWarning`](https://scikit-learn.org.cn/view/665.html) | 警告类，用于将行为的任何更改通知用户。 |
| --- | --- |
| [`exceptions.ConvergenceWarning`](https://scikit-learn.org.cn/view/666.html) | 自定义警告以捕获收敛问题 |
| [`exceptions.DataConversionWarning`](https://scikit-learn.org.cn/view/668.html) | 警告，用于通知代码中发生的隐式数据转换。 |
| [`exceptions.DataDimensionalityWarning`](https://scikit-learn.org.cn/view/669.html) | 自定义警告以通知潜在的数据维度问题。 |
| [`exceptions.EfficiencyWarning`](https://scikit-learn.org.cn/view/670.html) | 警告，用于通知用户计算效率低下。 |
| [`exceptions.FitFailedWarning`](https://scikit-learn.org.cn/view/671.html) | 如果在拟合估计器时发生错误，则使用警告类。 |
| [`exceptions.NotFittedError`](https://scikit-learn.org.cn/view/672.html) | 如果在拟合之前使用了估计量，则引发异常类。 |
| [`exceptions.NonBLASDotWarning`](https://scikit-learn.org.cn/view/674.html) | 点操作不使用BLAS时使用的警告。 |
| [`exceptions.UndefinedMetricWarning`](https://scikit-learn.org.cn/view/675.html) | 指标无效时使用的警告 |

## sklearn.experimental：实验

该[`sklearn.experimental`](https://scikit-learn.org.cn/lists/3.html#sklearn.experimental%EF%BC%9A%E5%AE%9E%E9%AA%8C)模块提供了可导入的模块，这些模块允许使用实验性功能或估算器。

实验性的功能和估计器不受弃用周期的限制。使用它们需要您自担风险！

| [`experimental.enable_hist_gradient_boosting`](https://scikit-learn.org.cn/view/677.html) | 启用基于直方图的梯度增强估计器。 |
| --- | --- |
| [`experimental.enable_iterative_imputer`](https://scikit-learn.org.cn/view/678.html) | 使迭代的输入 |

## sklearn.feature\_extraction特征提取

该[`sklearn.feature_extraction`](https://scikit-learn.org.cn/lists/3.html#sklearn.feature_extraction%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)模块负责从原始数据中提取特征。当前，它包括从文本和图像中提取特征的方法。

**用户指南**：有关更多详细信息，请参见[特征提取](https://scikit-learn.org.cn/view/122.html)部分。

| [`feature_extraction.DictVectorizer`](https://scikit-learn.org.cn/view/697.html)（\* \[，…\]） | 将特征值映射列表转换为矢量。 |
| --- | --- |
| [`feature_extraction.FeatureHasher`](https://scikit-learn.org.cn/view/699.html)（\[…\]） | 实现特征哈希，又名哈希技巧。 |

### 从图片

该`sklearn.feature_extraction.image`子模块收集实用程序以从图像中提取特征。

| [`feature_extraction.image.extract_patches_2d`](https://scikit-learn.org.cn/view/700.html)（...） | 将2D图像重塑为补丁集合 |
| --- | --- |
| [`feature_extraction.image.grid_to_graph`](https://scikit-learn.org.cn/view/702.html)（n\_x，n\_y） | 像素间连接图 |
| [`feature_extraction.image.img_to_graph`](https://scikit-learn.org.cn/view/703.html)（img，\*） | 像素间梯度连接图 |
| [`feature_extraction.image.reconstruct_from_patches_2d`](https://scikit-learn.org.cn/view/718.html)（...） | 从所有修补程序重建图像。 |
| [`feature_extraction.image.PatchExtractor`](https://scikit-learn.org.cn/view/720.html)（\* \[，…\]） | 从图像集合中提取补丁 |

### 从文字

该`sklearn.feature_extraction.text`子模块收集实用程序以从文本文档构建特征向量。

| [`feature_extraction.text.CountVectorizer`](https://scikit-learn.org.cn/view/723.html)（\* \[，…\]） | 将文本文档集合转换为令牌计数矩阵 |
| --- | --- |
| [`feature_extraction.text.HashingVectorizer`](https://scikit-learn.org.cn/view/726.html)（\*） | 将文本文档的集合转换为令牌出现的矩阵 |
| [`feature_extraction.text.TfidfTransformer`](https://scikit-learn.org.cn/view/727.html)（\*） | 将计数矩阵转换为标准化的tf或tf-idf表示形式 |
| [`feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org.cn/view/728.html)（\* \[，…\]） | 将原始文档集合转换为TF-IDF功能矩阵。 |

## sklearn.feature\_selection：特征选择

该[`sklearn.feature_selection`](https://scikit-learn.org.cn/lists/3.html#sklearn.feature_selection%EF%BC%9A%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9)模块实现特征选择算法。目前，它包括单变量过滤器选择方法和递归特征消除算法。

**用户指南**：有关更多详细信息，请参见“[特征选择](https://scikit-learn.org.cn/view/101.html)”部分。

| [`feature_selection.GenericUnivariateSelect`](https://scikit-learn.org.cn/view/734.html)（\[…\]） | 具有可配置策略的单变量特征选择器。 |
| --- | --- |
| [`feature_selection.SelectPercentile`](https://scikit-learn.org.cn/view/735.html)（\[…\]） | 根据最高分数的百分位数选择特征。 |
| [`feature_selection.SelectKBest`](https://scikit-learn.org.cn/view/737.html)（\[score\_func，k\]） | 根据k个最高分数选择特征。 |
| [`feature_selection.SelectFpr`](https://scikit-learn.org.cn/view/738.html)（\[score\_func，alpha\]） | 过滤器：根据FPR测试，在alpha以下选择p值。 |
| [`feature_selection.SelectFdr`](https://scikit-learn.org.cn/view/739.html)（\[score\_func，alpha\]） | 过滤器：为估计的错误发现率选择p值 |
| [`feature_selection.SelectFromModel`](https://scikit-learn.org.cn/view/741.html)(estimator, \*) | 元转换器，用于根据重要度选择特征。 |
| [`feature_selection.SelectFwe`](https://scikit-learn.org.cn/view/742.html)（\[score\_func，alpha\]） | 过滤器：选择与Family-wise错误率相对应的p值 |
| [`feature_selection.RFE`](https://scikit-learn.org.cn/view/743.html)(estimator, \*\[, …\]) | 消除递归特征的特征排名。 |
| [`feature_selection.RFECV`](https://scikit-learn.org.cn/view/745.html)(estimator, \*\[, …\]) | 通过消除递归特征和交叉验证最佳特征数选择来进行特征排名。 |
| [`feature_selection.VarianceThreshold`](https://scikit-learn.org.cn/view/749.html)(\[threshold\]) | 删除所有低方差特征的特征选择器。 |
| [`feature_selection.chi2`](https://scikit-learn.org.cn/view/750.html)（X，y） | 计算每个非负特征与类之间的卡方统计量。 |
| [`feature_selection.f_classif`](https://scikit-learn.org.cn/view/752.html)（X，y） | 计算提供的样本的ANOVA F值。 |
| [`feature_selection.f_regression`](https://scikit-learn.org.cn/view/754.html)（X，y，\* \[，中心\]） | 单变量线性回归测试。 |
| [`feature_selection.mutual_info_classif`](https://scikit-learn.org.cn/view/755.html)（X，y，\*） | 估计离散目标变量的互信息。 |
| [`feature_selection.mutual_info_regression`](https://scikit-learn.org.cn/view/756.html)（X，y，\*） | 估计一个连续目标变量的互信息。 |

## sklearn.gaussian\_process：高斯过程

该[`sklearn.gaussian_process`](https://scikit-learn.org.cn/lists/3.html#sklearn.gaussian_process%EF%BC%9A%E9%AB%98%E6%96%AF%E8%BF%87%E7%A8%8B)模块实现基于高斯过程的回归和分类。

**用户指南**：有关更多详细信息，请参见“ [高斯过程](https://scikit-learn.org.cn/view/86.html)”部分。

| [`gaussian_process.GaussianProcessClassifier`](https://scikit-learn.org.cn/view/859.html)（\[…\]） | 基于拉普拉斯近似的高斯过程分类（GPC）。 |
| --- | --- |
| [`gaussian_process.GaussianProcessRegressor`](https://scikit-learn.org.cn/view/862.html)（\[…\]） | 高斯过程回归（GPR）。 |

内核：

| [`gaussian_process.kernels.CompoundKernel`](https://scikit-learn.org.cn/view/863.html)（Kernel） | 由一组其他内核组成的内核。 |
| --- | --- |
| [`gaussian_process.kernels.ConstantKernel`](https://scikit-learn.org.cn/view/864.html)（\[…\]） | 恒定内核。 |
| [`gaussian_process.kernels.DotProduct`](https://scikit-learn.org.cn/view/865.html)（\[…\]） | 点积内核。 |
| [`gaussian_process.kernels.ExpSineSquared`](https://scikit-learn.org.cn/view/866.html)（\[…\]） | Exp-Sine-Squared核（也称为周期核）。 |
| [`gaussian_process.kernels.Exponentiation`](https://scikit-learn.org.cn/view/867.html)（...） | 幂运算内核采用一个基本内核和一个标量参数 p 并通过组合它们 |
| [`gaussian_process.kernels.Hyperparameter`](https://scikit-learn.org.cn/view/868.html) | 以命名元组形式表示的内核超参数规范。 |
| [`gaussian_process.kernels.Kernel`](https://scikit-learn.org.cn/view/869.html) | 所有内核的基类。 |
| [`gaussian_process.kernels.Matern`](https://scikit-learn.org.cn/view/870.html)（\[…\]） | 主内核。 |
| [`gaussian_process.kernels.PairwiseKernel`](https://scikit-learn.org.cn/view/871.html)（\[…\]） | sklearn.metrics.pairwise中的内核包装。 |
| [`gaussian_process.kernels.Product`](https://scikit-learn.org.cn/view/872.html)（k1，k2） | 该`Product`内核采用两个内核k1 和 k2 并通过组合它们 |
| [`gaussian_process.kernels.RBF`](https://scikit-learn.org.cn/view/873.html)（\[length\_scale，…\]） | 径向基函数内核（又名平方指数内核）。 |
| [`gaussian_process.kernels.RationalQuadratic`](https://scikit-learn.org.cn/view/874.html)（\[…\]） | 有理二次方内核。 |
| [`gaussian_process.kernels.Sum`](https://scikit-learn.org.cn/view/875.html)（k1，k2） | 该`Sum`内核采用两个内核k1 和 k2 并通过组合它们 |
| [`gaussian_process.kernels.WhiteKernel`](https://scikit-learn.org.cn/view/876.html)（\[…\]） | White kernel. |

## sklearn.impute：插补

缺失值估算的转换器

**用户指南**：有关更多详细信息，请参见[缺失值的插补](https://scikit-learn.org.cn/view/124.html)部分。

| [`impute.SimpleImputer`](https://scikit-learn.org.cn/view/763.html)（\* \[，missing\_values，…\]） | 插补转换器，用于填补缺失值。 |
| --- | --- |
| [`impute.IterativeImputer`](https://scikit-learn.org.cn/view/767.html)(\[estimator, …\]) | 从所有其他特征中估计每个特征的多元插补器。 |
| [`impute.MissingIndicator`](https://scikit-learn.org.cn/view/769.html)（\* \[，missing\_values，…\]） | 缺失值的二进制指标。 |
| [`impute.KNNImputer`](https://scikit-learn.org.cn/view/770.html)（\* \[，missing\_values，…\]） | 用k近邻填充缺失值。 |

## sklearn.inspection：检查

该[`sklearn.inspection`](https://scikit-learn.org.cn/lists/3.html#sklearn.inspection%EF%BC%9A%E6%A3%80%E6%9F%A5)模块包括用于模型检查的工具。

| [`inspection.partial_dependence`](https://scikit-learn.org.cn/view/847.html)(estimator, X, …) | `features`的部分依赖。 |
| --- | --- |
| [`inspection.permutation_importance`](https://scikit-learn.org.cn/view/849.html)(estimator, …) | 特征评价中的置换重要性\[[Rd9e56ef97513-BRE\]](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#rd9e56ef97513-bre)。 |

### 绘图

| [`inspection.PartialDependenceDisplay`](https://scikit-learn.org.cn/view/850.html)（...） | 部分依赖图（PDP）可视化。 |
| --- | --- |
| [`inspection.plot_partial_dependence`](https://scikit-learn.org.cn/view/851.html)（…\[，…\]） | 部分依赖图。 |

## sklearn.isotonic：等渗回归

**用户指南**：有关更多详细信息，请参见“[等渗回归](https://scikit-learn.org.cn/lists/3.html#sklearn.isotonic%EF%BC%9A%E7%AD%89%E6%B8%97%E5%9B%9E%E5%BD%92)”部分。

| [`isotonic.IsotonicRegression`](https://scikit-learn.org.cn/view/852.html)（\* \[，y\_min，…\]） | 等渗回归模型。 |
| --- | --- |
| [`isotonic.check_increasing`](https://scikit-learn.org.cn/view/853.html)（x，y） | 确定y是否与x单调相关。 |
| [`isotonic.isotonic_regression`](https://scikit-learn.org.cn/view/854.html)（y，\* \[，…\]） | 求解等渗回归模型。 |

## sklearn.kernel\_approximation内核近似

该[`sklearn.kernel_approximation`](https://scikit-learn.org.cn/lists/3.html#sklearn.kernel_approximation%E5%86%85%E6%A0%B8%E8%BF%91%E4%BC%BC)模块基于傅立叶变换实现了几个近似的内核特征图。

**用户指南**：有关更多详细信息，请参见“[内核近似](https://scikit-learn.org.cn/view/127.html)”部分。

| [`kernel_approximation.AdditiveChi2Sampler`](https://scikit-learn.org.cn/view/372.html)（\*） | chi2内核的近似特征图。 |
| --- | --- |
| [`kernel_approximation.Nystroem`](https://scikit-learn.org.cn/view/373.html)(\[kernel, …\]) | 使用训练数据的子集近似核图。 |
| [`kernel_approximation.RBFSampler`](https://scikit-learn.org.cn/view/374.html)（\* \[，gamma，…\]） | 通过傅立叶变换的蒙特卡洛近似来近似RBF内核的特征图。 |
| [`kernel_approximation.SkewedChi2Sampler`](https://scikit-learn.org.cn/view/376.html)（\* \[，…\]） | 通过傅立叶变换的蒙特卡洛近似来近似“倾斜的卡方”核的特征图。 |

## sklearn.kernel\_ridge内核岭回归

模块[`sklearn.kernel_ridge`](https://scikit-learn.org.cn/lists/3.html#sklearn.kernel_ridge%E5%86%85%E6%A0%B8%E5%B2%AD%E5%9B%9E%E5%BD%92)实现内核岭回归。

**用户指南**：有关更多详细信息，请参见“ [内核岭回归](https://scikit-learn.org.cn/view/80.html)”部分。

| [`kernel_ridge.KernelRidge`](https://scikit-learn.org.cn/view/377.html)（\[alpha，kernel，...\]） |内核岭回归。 |

## sklearn.linear\_model：线性模型

该[`sklearn.linear_model`](https://scikit-learn.org.cn/lists/3.html#sklearn.linear_model%EF%BC%9A%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B)模块实现了各种线性模型。

**用户指南**：有关更多详细信息，请参见“ [线性模型](https://scikit-learn.org.cn/view/4.html)”部分。

以下小节仅是粗略的指导原则：相同的估算器可以根据其参数分为多个类别。

### 线性分类

| [`linear_model.LogisticRegression`](https://scikit-learn.org.cn/view/378.html)(\[penalty, …\]) | Logistic回归（又名logit，MaxEnt）分类器。 |
| --- | --- |
| [`linear_model.LogisticRegressionCV`](https://scikit-learn.org.cn/view/381.html)（\* \[，Cs，…\]） | Logistic回归CV（又名logit，MaxEnt）分类器。 |
| [`linear_model.PassiveAggressiveClassifier`](https://scikit-learn.org.cn/view/382.html)（\*） | 被动感知分类器 |
| [`linear_model.Perceptron`](https://scikit-learn.org.cn/view/384.html)(\*\[, penalty, alpha, …\]) | 在《[用户指南](https://scikit-learn.org.cn/lists/2.html)》中阅读更多内容。 |
| [`linear_model.RidgeClassifier`](https://scikit-learn.org.cn/view/385.html)（\[α， …\]） | 使用Ridge回归的分类器。 |
| [`linear_model.RidgeClassifierCV`](https://scikit-learn.org.cn/view/386.html)（\[alphas，...\]） | 带有内置交叉验证的Ridge分类器。 |
| [`linear_model.SGDClassifier`](https://scikit-learn.org.cn/view/388.html)(\[loss, penalty, …\]) | 具有SGD训练的线性分类器（SVM，逻辑回归等）。 |

### 经典线性回归器

| [`linear_model.LinearRegression`](https://scikit-learn.org.cn/view/394.html)（\* \[，…\]） | 普通最小二乘线性回归。 |
| --- | --- |
| [`linear_model.Ridge`](https://scikit-learn.org.cn/view/399.html)（\[alpha，fit\_intercept，…\]） | 具有l2正则化的线性最小二乘法。 |
| [`linear_model.RidgeCV`](https://scikit-learn.org.cn/view/400.html)（\[alphas，...\]） | 带有内置交叉验证的Ridge回归。 |
| [`linear_model.SGDRegressor`](https://scikit-learn.org.cn/view/401.html)(\[loss, penalty, …\]) | 通过使用SGD最小化正则经验损失来拟合线性模型 |

### 具有特征选择的回归器

以下估计器具有内置的特征选择拟合程序，但是任何使用L1或弹性网惩罚的估计器也将执行特征选择：通常`SGDRegressor` 或`SGDClassifier`具有适当的罚分。

| [`linear_model.ElasticNet`](https://scikit-learn.org.cn/view/404.html)（\[alpha，l1\_ratio，…\]） | 将L1和L2先验组合作为正则化器的线性回归。 |
| --- | --- |
| [`linear_model.ElasticNetCV`](https://scikit-learn.org.cn/view/407.html)（\* \[，l1\_ratio，…\]） | 沿着正则化路径具有迭代拟合的弹性网模型。 |
| [`linear_model.Lars`](https://scikit-learn.org.cn/view/409.html)（\* \[，fit\_intercept，…\]） | 最小角度回归模型。 |
| [`linear_model.LarsCV`](https://scikit-learn.org.cn/view/410.html)（\* \[，fit\_intercept，…\]） | 交叉验证的最小角度回归模型。 |
| [`linear_model.Lasso`](https://scikit-learn.org.cn/view/411.html)（\[alpha，fit\_intercept，…\]） | 以L1先验作为正则化器训练的线性模型(又名套索) |
| [`linear_model.LassoCV`](https://scikit-learn.org.cn/view/414.html)（\* \[，eps，n\_alphas，…\]） | 沿正则化路径迭代拟合的套索线性模型。 |
| [`linear_model.LassoLars`](https://scikit-learn.org.cn/view/416.html)（\[α， …\]） | 套索模型与最小角度回归拟合 |
| [`linear_model.LassoLarsCV`](https://scikit-learn.org.cn/view/417.html)（\* \[，fit\_intercept，…\]） | 使用LARS算法进行交叉验证的套索。 |
| [`linear_model.LassoLarsIC`](https://scikit-learn.org.cn/view/418.html)(\[criterion, …\]) | 使用BIC或AIC选择模型的套索模型与Lars拟合 |
| [`linear_model.OrthogonalMatchingPursuit`](https://scikit-learn.org.cn/view/419.html)（\* \[，…\]） | 正交匹配追踪模型（OMP） |
| [`linear_model.OrthogonalMatchingPursuitCV`](https://scikit-learn.org.cn/view/420.html)（\*） | 交叉验证的正交匹配追踪模型（OMP）。 |

### 贝叶斯回归器

| [`linear_model.ARDRegression`](https://scikit-learn.org.cn/view/421.html)（\* \[，n\_iter，tol，…\]） | 贝叶斯ARD回归。 |
| --- | --- |
| [`linear_model.BayesianRidge`](https://scikit-learn.org.cn/view/422.html)（\* \[，n\_iter，tol，…\]） | 贝叶斯岭回归。 |

### 具有特征选择的多任务线性回归器

这些估计器共同拟合多个回归问题（或任务），同时得出稀疏系数。尽管推断的系数在任务之间可能有所不同，但它们被约束为在选定的特征（非零系数）上达成一致。

| [`linear_model.MultiTaskElasticNet`](https://scikit-learn.org.cn/view/425.html)（\[α， …\]） | 以L1 / L2混合范数为正则训练的多任务弹性网模型 |
| --- | --- |
| [`linear_model.MultiTaskElasticNetCV`](https://scikit-learn.org.cn/view/426.html)（\* \[，…\]） | 具有内置交叉验证的多任务L1 / L2 弹性网。 |
| [`linear_model.MultiTaskLasso`](https://scikit-learn.org.cn/view/427.html)（\[α， …\]） | 以L1 / L2混合范数为正则训练的多任务套索模型。 |
| [`linear_model.MultiTaskLassoCV`](https://scikit-learn.org.cn/view/428.html)（\* \[，eps，…\]） | 以L1 / L2混合范数为正则训练的带有交叉验证的多任务套索模型。 |

### 异常值稳健回归器

使用Huber损失的任何估计量也将对异常值具有鲁棒性，例如 [`SGDRegressor`](https://scikit-learn.org.cn/view/401.html)使用`loss='huber'`。

| [`linear_model.HuberRegressor`](https://scikit-learn.org.cn/view/430.html)（\* \[，epsilon，…\]） | 对异常值具有鲁棒性的线性回归模型。 |
| --- | --- |
| [`linear_model.RANSACRegressor`](https://scikit-learn.org.cn/view/431.html)（\[…\]） | RANSAC（随机抽样共识）算法。 |
| [`linear_model.TheilSenRegressor`](https://scikit-learn.org.cn/view/433.html)（\* \[，…\]） | Theil-Sen估算器：稳健的多元回归模型。 |

### 广义线性回归模型（GLM）

这些模型允许响应变量具有除正态分布之外的其他误差分布：

| [`linear_model.PoissonRegressor`](https://scikit-learn.org.cn/view/437.html)（\*\[， α， …\]） | 具有泊松分布的广义线性模型。 |
| --- | --- |
| [`linear_model.TweedieRegressor`](https://scikit-learn.org.cn/view/439.html)(\*\[, power, …\]) | 具有Tweedie分布的广义线性模型。 |
| [`linear_model.GammaRegressor`](https://scikit-learn.org.cn/view/440.html)（\*\[， α， …\]） | 具有Gamma分布的广义线性模型。 |

### 杂项

| [`linear_model.PassiveAggressiveRegressor`](https://scikit-learn.org.cn/view/442.html)（\* \[，…\]） | 被动感知回归 |
| --- | --- |
| [`linear_model.enet_path`](https://scikit-learn.org.cn/view/443.html)（X，y，\* \[，l1\_ratio，…\]） | 用坐标下降计算弹性网路径。 |
| [`linear_model.lars_path`](https://scikit-learn.org.cn/view/445.html)（X，y \[，Xy，Gram，…\]） | 使用LARS算法计算最小角度回归或套索路径\[1\] |
| [`linear_model.lars_path_gram`](https://scikit-learn.org.cn/view/446.html)（Xy，Gram，\*，…） | 统计模式下的lars\_path \[1\] |
| [`linear_model.lasso_path`](https://scikit-learn.org.cn/view/447.html)（X，y，\* \[，eps，…\]） | 计算具有坐标下降的套索路径 |
| [`linear_model.orthogonal_mp`](https://scikit-learn.org.cn/view/448.html)（X，y，\* \[，…\]） | 正交匹配追踪（OMP） |
| [`linear_model.orthogonal_mp_gram`](https://scikit-learn.org.cn/view/449.html)（Gram，Xy，\*） | 伽马正交匹配追踪（OMP） |
| [`linear_model.ridge_regression`](https://scikit-learn.org.cn/view/450.html)（X，y，alpha，\*） | 用正规方程法求解岭方程。 |

## sklearn.manifold：流形学习

该[`sklearn.manifold`](https://scikit-learn.org.cn/lists/3.html#sklearn.manifold%EF%BC%9A%E6%B5%81%E5%BD%A2%E5%AD%A6%E4%B9%A0)模块实现数据嵌入技术。

**用户指南**：有关更多详细信息，请参见“[流形学习](https://scikit-learn.org.cn/view/107.html)”部分。

| [`manifold.Isomap`](https://scikit-learn.org.cn/view/452.html)（\* \[，n\_neighbors，…\]） | 等值图嵌入 |
| --- | --- |
| [`manifold.LocallyLinearEmbedding`](https://scikit-learn.org.cn/view/455.html)（\* \[，…\]） | 局部线性嵌入 |
| [`manifold.MDS`](https://scikit-learn.org.cn/view/459.html)(\[n\_components, metric, n\_init, …\]) | 多维缩放 |
| [`manifold.SpectralEmbedding`](https://scikit-learn.org.cn/view/461.html)（\[n\_components，...\]） | 频谱嵌入用于非线性降维。 |
| [`manifold.TSNE`](https://scikit-learn.org.cn/view/463.html)(\[n\_components, perplexity, …\]) | t分布随机邻接嵌入。 |
| [`manifold.locally_linear_embedding`](https://scikit-learn.org.cn/view/464.html)（X， \*， …） | 对数据执行局部线性嵌入分析。 |
| [`manifold.smacof`](https://scikit-learn.org.cn/view/465.html)(dissimilarities, \*\[, …\]) | 使用SMACOF算法计算多维缩放。 |
| [`manifold.spectral_embedding`](https://scikit-learn.org.cn/view/467.html)(adjacency, \*\[, …\]) | 将样本投影到图拉普拉斯算子的第一个特征向量上。 |
| [`manifold.trustworthiness`](https://scikit-learn.org.cn/view/468.html)（X，X\_embedded，\* \[，…\]） | 表示保留本地结构的程度。 |

## sklearn.metrics：指标

有关更多详细信息，请参阅用户指南的“[指标和评分：量化预测的质量](https://scikit-learn.org.cn/view/93.html)”部分和“[成对度量，近似关系和内核](https://scikit-learn.org.cn/view/128.html)”部分。

该[`sklearn.metrics`](https://scikit-learn.org.cn/lists/3.html#sklearn.metrics%EF%BC%9A%E6%8C%87%E6%A0%87)模块包括评分功能，性能指标以及成对指标和距离计算。

### 选型界面

有关更多详细信息，请参见用户指南的“[评分参数：定义模型评估规则](http://scikit-learn.jg.com.cn/view/93.html#3.3.1%20%E8%AF%84%E5%88%86%E5%8F%82%E6%95%B0%EF%BC%9A%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0%E5%87%86%E5%88%99)”部分。

| [`metrics.check_scoring`](https://scikit-learn.org.cn/view/471.html)(estimator\[, scoring, …\]) | 从用户选项确定计分器。 |
| --- | --- |
| [`metrics.get_scorer`](https://scikit-learn.org.cn/view/472.html)（得分） | 从字符串中获取一个得分手。 |
| [`metrics.make_scorer`](https://scikit-learn.org.cn/view/474.html)（score\_func，\* \[，…\]） | 根据绩效指标或损失函数确定得分手。 |

### 分类指标

有关更多详细信息，请参见用户指南的“ [分类指标](https://scikit-learn.org.cn/view/93.html#3.3.2%20%E5%88%86%E7%B1%BB%E6%8C%87%E6%A0%87)”部分。

| [`metrics.accuracy_score`](https://scikit-learn.org.cn/view/476.html)（y\_true，y\_pred，\* \[，…\]） | 精度分类得分。 |
| --- | --- |
| [`metrics.auc`](https://scikit-learn.org.cn/view/478.html)（x，y） | 使用梯形法则计算曲线下面积（AUC） |
| [`metrics.average_precision_score`](https://scikit-learn.org.cn/view/479.html)（y\_true，...） | 根据预测分数计算平均精度（AP） |
| [`metrics.balanced_accuracy_score`](https://scikit-learn.org.cn/view/480.html)（y\_true，...） | 计算平衡精度 |
| [`metrics.brier_score_loss`](https://scikit-learn.org.cn/view/481.html)（y\_true，y\_prob，\*） | 计算Brier分数。 |
| [`metrics.classification_report`](https://scikit-learn.org.cn/view/482.html)（y\_true，y\_pred，\*） | 建立一个显示主要分类指标的文本报告。 |
| [`metrics.cohen_kappa_score`](https://scikit-learn.org.cn/view/484.html)（y1，y2，\* \[，...\]） | 科恩的kappa：一种用于度量注释者之间协议的统计数据。 |
| [`metrics.confusion_matrix`](https://scikit-learn.org.cn/view/485.html)（y\_true，y\_pred，\*） | 计算混淆矩阵以评估分类的准确性。 |
| [`metrics.dcg_score`](https://scikit-learn.org.cn/view/486.html)（y\_true，y\_score，\* \[，k，...\]） | 计算折现累积收益。 |
| [`metrics.f1_score`](https://scikit-learn.org.cn/view/487.html)（y\_true，y\_pred，\* \[，…\]） | 计算F1分数，也称为平衡F分数或F测量 |
| [`metrics.fbeta_score`](https://scikit-learn.org.cn/view/488.html)（y\_true，y\_pred，\*，beta） | 计算F-beta分数 |
| [`metrics.hamming_loss`](https://scikit-learn.org.cn/view/489.html)（y\_true，y\_pred，\* \[，…\]） | 计算平均汉明损失。 |
| [`metrics.hinge_loss`](https://scikit-learn.org.cn/view/490.html)（y\_true，pred\_decision，\*） | 平均铰链损耗（非常规） |
| [`metrics.jaccard_score`](https://scikit-learn.org.cn/view/491.html)（y\_true，y\_pred，\* \[，…\]） | 雅卡德相似系数得分 |
| [`metrics.log_loss`](https://scikit-learn.org.cn/view/492.html)（y\_true，y\_pred，\* \[，eps，…\]） | 对数损失，aka逻辑损失或交叉熵损失。 |
| [`metrics.matthews_corrcoef`](https://scikit-learn.org.cn/view/493.html)（y\_true，y\_pred，\*） | 计算马修斯相关系数（MCC） |
| [`metrics.multilabel_confusion_matrix`](https://scikit-learn.org.cn/view/494.html)（y\_true，...） | 为每个类别或样本计算混淆矩阵 |
| [`metrics.ndcg_score`](https://scikit-learn.org.cn/view/495.html)（y\_true，y\_score，\* \[，k，...\]） | 计算归一化折现累积增益。 |
| [`metrics.precision_recall_curve`](https://scikit-learn.org.cn/view/496.html)（y\_true，...） | 计算不同概率阈值的精确召回对 |
| [`metrics.precision_recall_fscore_support`](https://scikit-learn.org.cn/view/497.html)（...） | 计算每个班级的精度，召回率，F量度和支持 |
| [`metrics.precision_score`](https://scikit-learn.org.cn/view/498.html)（y\_true，y\_pred，\* \[，…\]） | 计算精度 |
| [`metrics.recall_score`](https://scikit-learn.org.cn/view/499.html)（y\_true，y\_pred，\* \[，…\]） | 计算召回率 |
| [`metrics.roc_auc_score`](https://scikit-learn.org.cn/view/500.html)（y\_true，y\_score，\* \[，…\]） | 根据预测分数计算接收器工作特性曲线（ROC AUC）下的面积。 |
| [`metrics.roc_curve`](https://scikit-learn.org.cn/view/501.html)（y\_true，y\_score，\* \[，…\]） | 计算接收器工作特性（ROC） |
| [`metrics.zero_one_loss`](https://scikit-learn.org.cn/view/502.html)（y\_true，y\_pred，\* \[，…\]） | 零一分类损失。 |

### 回归指标

有关更多详细信息，请参见用户指南的"[回归指标](https://scikit-learn.org.cn/view/93.html#3.3.4%20%E5%9B%9E%E5%BD%92%E6%8C%87%E6%A0%87)"部分。

| [`metrics.explained_variance_score`](https://scikit-learn.org.cn/view/509.html)（y\_true，...） | 解释方差回归得分函数 |
| --- | --- |
| [`metrics.max_error`](https://scikit-learn.org.cn/view/510.html)（y\_true，y\_pred） | max\_error指标计算最大残差。 |
| [`metrics.mean_absolute_error`](https://scikit-learn.org.cn/view/511.html)（y\_true，y\_pred，\*） | 平均绝对误差回归损失 |
| [`metrics.mean_squared_error`](https://scikit-learn.org.cn/view/513.html)（y\_true，y\_pred，\*） | 均方误差回归损失 |
| [`metrics.mean_squared_log_error`](https://scikit-learn.org.cn/view/516.html)（y\_true，y\_pred，\*） | 均方对数误差回归损失 |
| [`metrics.median_absolute_error`](https://scikit-learn.org.cn/view/517.html)（y\_true，y\_pred，\*） | 中值绝对误差回归损失 |
| [`metrics.r2_score`](https://scikit-learn.org.cn/view/519.html)（y\_true，y\_pred，\* \[，…\]） | R ^ 2（确定系数）回归得分函数。 |
| [`metrics.mean_poisson_deviance`](https://scikit-learn.org.cn/view/521.html)（y\_true，y\_pred，\*） | 平均泊松偏差回归损失。 |
| [`metrics.mean_gamma_deviance`](https://scikit-learn.org.cn/view/522.html)（y\_true，y\_pred，\*） | 平均伽玛偏差回归损失。 |
| [`metrics.mean_tweedie_deviance`](https://scikit-learn.org.cn/view/523.html)（y\_true，y\_pred，\*） | 平均Tweedie偏差回归损失。 |

### 多标签排名指标

有关更多详细信息，请参见用户指南的“ [多标签排名指标](https://scikit-learn.org.cn/view/93.html)”部分。

| [`metrics.coverage_error`](https://scikit-learn.org.cn/view/524.html)（y\_true，y\_score，\* \[，…\]） | 覆盖误差测量 |
| --- | --- |
| [`metrics.label_ranking_average_precision_score`](https://scikit-learn.org.cn/view/525.html)（...） | 计算基于排名的平均精度 |
| [`metrics.label_ranking_loss`](https://scikit-learn.org.cn/view/526.html)（y\_true，y\_score，\*） | 计算排名损失度量 |

### 聚类指标

有关更多详细信息，请参见用户指南的“ [聚类性能评估](https://scikit-learn.org.cn/view/108.html#2.3.10.%20%E8%81%9A%E7%B1%BB%E6%80%A7%E8%83%BD%E5%BA%A6%E9%87%8F)”部分。

该`sklearn.metrics.cluster`子模块包含用于聚类分析结果的评估指标。评估有两种形式：

-   监督，它为每个样本使用基本事实类别值。
    
-   无监督的，它不会并且无法衡量模型本身的“质量”。
    

| [`metrics.adjusted_mutual_info_score`](https://scikit-learn.org.cn/view/527.html)（…\[，…\]） | 调整两个簇之间的相互信息。 |
| --- | --- |
| [`metrics.adjusted_rand_score`](https://scikit-learn.org.cn/view/528.html)（labels\_true，...） | 经过调整的兰德指数。 |
| [`metrics.calinski_harabasz_score`](https://scikit-learn.org.cn/view/529.html)(X, labels) | 计算Calinski和Harabasz得分。 |
| [`metrics.davies_bouldin_score`](https://scikit-learn.org.cn/view/530.html)(X, labels) | 计算Davies-Bouldin分数。 |
| [`metrics.completeness_score`](https://scikit-learn.org.cn/view/531.html)（labels\_true，...） | 给定真值的聚类标记的完备性度量。 |
| [`metrics.cluster.contingency_matrix`](https://scikit-learn.org.cn/view/533.html)（…\[，…\]） | 建立一个列联矩阵来描述标签之间的关系。 |
| [`metrics.fowlkes_mallows_score`](https://scikit-learn.org.cn/view/534.html)（labels\_true，...） | 度量一组点的两个簇的相似性。 |
| [`metrics.homogeneity_completeness_v_measure`](https://scikit-learn.org.cn/view/535.html)（...） | 一次计算同质性和完整性以及V-Measure分数。 |
| [`metrics.homogeneity_score`](https://scikit-learn.org.cn/view/536.html)（labels\_true，...） | 给定真值的聚类标记的同质性度量。 |
| [`metrics.mutual_info_score`](https://scikit-learn.org.cn/view/537.html)（labels\_true，...） | 两个簇之间的相互信息。 |
| [`metrics.normalized_mutual_info_score`](https://scikit-learn.org.cn/view/538.html)（…\[，…\]） | 两个簇之间的标准化互信息。 |
| [`metrics.silhouette_score`](https://scikit-learn.org.cn/view/539.html)(X, labels, \*\[, …\]) | 计算所有样本的平均轮廓系数。 |
| [`metrics.silhouette_samples`](https://scikit-learn.org.cn/view/541.html)(X, labels, \*\[, …\]) | 计算每个样本的轮廓系数。 |
| [`metrics.v_measure_score`](https://scikit-learn.org.cn/view/544.html)（labels\_true，…\[，beta\]） | 给定一个真值的V-度量聚类标记。 |

### 分类指标

有关更多详细信息，请参见用户指南的"[分类评估](https://scikit-learn.org.cn/view/109.html#2.4.3.%20Biclustering%20%E8%AF%84%E4%BB%B7)"部分。

| [`metrics.consensus_score`](https://scikit-learn.org.cn/view/550.html)(a, b, \*\[, similarity\]) | 两个簇的相似性。 |

### 成对指标

有关更多详细信息，请参见用户指南的"[成对度量，近似关系和内核](https://scikit-learn.org.cn/view/128.html)"部分。

| [`metrics.pairwise.additive_chi2_kernel`](https://scikit-learn.org.cn/view/551.html)（X \[，Y\]） | 计算X和Y观测值之间的加性方卡方核 |
| --- | --- |
| [`metrics.pairwise.chi2_kernel`](https://scikit-learn.org.cn/view/553.html)（X \[，Y，γ） | 计算指数卡方内核X和Y。 |
| [`metrics.pairwise.cosine_similarity`](https://scikit-learn.org.cn/view/554.html)（X \[，Y，…\]） | 计算X和Y中样本之间的余弦相似度。 |
| [`metrics.pairwise.cosine_distances`](https://scikit-learn.org.cn/view/555.html)（X \[，Y\]） | 计算X和Y中样本之间的余弦距离。 |
| [`metrics.pairwise.distance_metrics`](https://scikit-learn.org.cn/view/557.html)（） | pairwise\_distances的有效指标。 |
| [`metrics.pairwise.euclidean_distances`](https://scikit-learn.org.cn/view/558.html)（X \[，Y，…\]） | 将X（和Y = X）的行视为向量，计算每对向量之间的距离矩阵。 |
| [`metrics.pairwise.haversine_distances`](https://scikit-learn.org.cn/view/559.html)（X \[，Y\]） | 计算X和Y中样本之间的Haversine距离 |
| [`metrics.pairwise.kernel_metrics`](https://scikit-learn.org.cn/view/560.html)（） | pairwise\_kernels的有效指标 |
| [`metrics.pairwise.laplacian_kernel`](https://scikit-learn.org.cn/view/561.html)（X \[，Y，γ） | 计算X和Y之间的拉普拉斯核。 |
| [`metrics.pairwise.linear_kernel`](https://scikit-learn.org.cn/view/562.html)（X \[，Y，…\]） | 计算X和Y之间的线性核。 |
| [`metrics.pairwise.manhattan_distances`](https://scikit-learn.org.cn/view/563.html)（X \[，Y，…\]） | 计算X和Y中向量之间的L1距离。 |
| [`metrics.pairwise.nan_euclidean_distances`](https://scikit-learn.org.cn/view/564.html)（X） | 在缺少值的情况下计算欧几里得距离。 |
| [`metrics.pairwise.pairwise_kernels`](https://scikit-learn.org.cn/view/565.html)（X \[，Y，…\]） | 计算数组X和可选数组Y之间的内核。 |
| [`metrics.pairwise.polynomial_kernel`](ttps://scikit-learn.org.cn/view/566.html)（X \[，Y，…\]） | 计算X和Y之间的多项式核。 |
| [`metrics.pairwise.rbf_kernel`](https://scikit-learn.org.cn/view/567.html)（X \[，Y，γ） | 计算X和Y之间的rbf（高斯）内核。 |
| [`metrics.pairwise.sigmoid_kernel`](https://scikit-learn.org.cn/view/568.html)（X \[，Y，…\]） | 计算X和Y之间的S形核。 |
| [`metrics.pairwise.paired_euclidean_distances`](https://scikit-learn.org.cn/view/569.html)（X，Y） | 计算X和Y之间的成对的欧式距离 |
| [`metrics.pairwise.paired_manhattan_distances`](https://scikit-learn.org.cn/view/570.html)（X，Y） | 计算X和Y中向量之间的L1距离。 |
| [`metrics.pairwise.paired_cosine_distances`](https://scikit-learn.org.cn/view/572.html)（X，Y） | 计算X和Y之间的配对余弦距离 |
| [`metrics.pairwise.paired_distances`](https://scikit-learn.org.cn/view/573.html)（X，Y，\* \[，…\]） | 计算X和Y之间的配对距离。 |
| [`metrics.pairwise_distances`](https://scikit-learn.org.cn/view/574.html)（X \[，Y，metric，…\]） | 根据向量数组X和可选的Y计算距离矩阵。 |
| [`metrics.pairwise_distances_argmin`](https://scikit-learn.org.cn/view/576.html)（X，Y，\* \[，…\]） | 计算一个点与一组点之间的最小距离。 |
| [`metrics.pairwise_distances_argmin_min`](https://scikit-learn.org.cn/view/577.html)（X，Y，\*） | 计算一个点与一组点之间的最小距离。 |
| [`metrics.pairwise_distances_chunked`](https://scikit-learn.org.cn/view/578.html)（X \[，Y，…\]） | 通过可选缩减逐块生成距离矩阵 |

### 绘图

有关更多详细信息，请参见用户指南的“ [可视化](https://scikit-learn.org.cn/view/95.html)”部分。

| [`metrics.plot_confusion_matrix`](https://scikit-learn.org.cn/view/579.html)(estimator, X, …) | 绘制混淆矩阵。 |
| --- | --- |
| [`metrics.plot_precision_recall_curve`](https://scikit-learn.org.cn/view/580.html)（…\[，…\]） | 绘制二元分类器的精确召回曲线。 |
| [`metrics.plot_roc_curve`](https://scikit-learn.org.cn/view/581.html)(estimator, X, y, \*\[, …\]) | 绘制接收器工作特性（ROC）曲线。 |
| [`metrics.ConfusionMatrixDisplay`](https://scikit-learn.org.cn/view/582.html)（…\[，…\]） | 混淆矩阵可视化。 |
| [`metrics.PrecisionRecallDisplay`](https://scikit-learn.org.cn/view/584.html)(precision, …) | 精确调用可视化。 |
| [`metrics.RocCurveDisplay`](https://scikit-learn.org.cn/view/585.html)（\*，fpr，tpr \[，…\]） | ROC曲线可视化。 |

## sklearn.mixture：高斯混合模型

该[`sklearn.mixture`](https://scikit-learn.org.cn/lists/3.html#sklearn.mixture%EF%BC%9A%E9%AB%98%E6%96%AF%E6%B7%B7%E5%90%88%E6%A8%A1%E5%9E%8B)模块实现了混合建模算法。

**用户指南**：有关更多详细信息，请参见“ [高斯混合模型](https://scikit-learn.org.cn/view/106.html)”部分。

| [`mixture.BayesianGaussianMixture`](https://scikit-learn.org.cn/view/630.html)（\* \[，…\]） | 高斯混合的变分贝叶斯估计。 |
| --- | --- |
| [`mixture.GaussianMixture`](https://scikit-learn.org.cn/view/632.html)（\[n\_components，...\]） | 高斯混合。 |

## sklearn.model\_selection：模型选择

**用户指南**：请参阅[交叉验证：评估模型表现](https://scikit-learn.org.cn/view/6.html)，[调整](https://scikit-learn.org.cn/view/99.html)[估计器](https://scikit-learn.org.cn/view/6.html)[的超参数](https://scikit-learn.org.cn/view/99.html)和 [学习曲线](https://scikit-learn.org.cn/view/116.html)部分，以了解更多详细信息。

### 拆分器类

| [`model_selection.GroupKFold`](https://scikit-learn.org.cn/view/634.html)（\[n\_splits\]） | 具有非重叠组的K折叠迭代器变体。 |
| --- | --- |
| [`model_selection.GroupShuffleSplit`](https://scikit-learn.org.cn/view/635.html)（\[…\]） | 随机分组交叉验证迭代器 |
| [`model_selection.KFold`](https://scikit-learn.org.cn/view/636.html)(\[n\_splits, shuffle, …\]) | K折交叉验证器 |
| [`model_selection.LeaveOneGroupOut`](https://scikit-learn.org.cn/view/637.html) | 离开一个小组的交叉验证者 |
| [`model_selection.LeavePGroupsOut`](https://scikit-learn.org.cn/view/638.html)（n\_groups） | 保留P组交叉验证器 |
| [`model_selection.LeaveOneOut`](https://scikit-learn.org.cn/view/639.html) | 留一法交叉验证器 |
| [`model_selection.LeavePOut`](https://scikit-learn.org.cn/view/640.html)（p） | Leave-P-Out交叉验证器 |
| [`model_selection.PredefinedSplit`](https://scikit-learn.org.cn/view/641.html)（test\_fold） | 预定义的拆分交叉验证器 |
| [`model_selection.RepeatedKFold`](https://scikit-learn.org.cn/view/642.html)（\* \[，n\_splits，…\]） | 重复的K折交叉验证器。 |
| [`model_selection.RepeatedStratifiedKFold`](https://scikit-learn.org.cn/view/643.html)（\* \[，…\]） | 重复分层K折交叉验证器。 |
| [`model_selection.ShuffleSplit`](https://scikit-learn.org.cn/view/644.html)（\[n\_splits，...\]） | 随机置换交叉验证器 |
| [`model_selection.StratifiedKFold`](https://scikit-learn.org.cn/view/645.html)（\[n\_splits，...\]） | 分层K折交叉验证器 |
| [`model_selection.StratifiedShuffleSplit`](https://scikit-learn.org.cn/view/646.html)（\[…\]） | 分层ShuffleSplit交叉验证器 |
| [`model_selection.TimeSeriesSplit`](https://scikit-learn.org.cn/view/647.html)（\[n\_splits，...\]） | 时间序列交叉验证器 |

### 拆分器函数

| [`model_selection.check_cv`](https://scikit-learn.org.cn/view/648.html)(\[cv, y, classifier\]) | 输入检查器实用程序，用于构建交叉验证器 |
| --- | --- |
| [`model_selection.train_test_split`](https://scikit-learn.org.cn/view/649.html)(\*arrays, …) | 将数组或矩阵拆分为随机训练和测试子集 |

### 超参数优化器

| [`model_selection.GridSearchCV`](https://scikit-learn.org.cn/view/655.html)(estimator, …) | 详尽搜索估计器的指定参数值。 |
| --- | --- |
| [`model_selection.ParameterGrid`](https://scikit-learn.org.cn/view/656.html)（param\_grid） | 参数的网格，每个网格都有离散数量的值。 |
| [`model_selection.ParameterSampler`](https://scikit-learn.org.cn/view/657.html)（…\[，…\]） | 根据给定分布采样的参数生成器。 |
| [`model_selection.RandomizedSearchCV`](https://scikit-learn.org.cn/view/658.html)（…\[，…\]） | 随机搜索超参数。 |

### 模型验证

| [`model_selection.cross_validate`](https://scikit-learn.org.cn/view/660.html)(estimator, X) | 通过交叉验证评估指标，并记录拟合/得分时间。 |
| --- | --- |
| [`model_selection.cross_val_predict`](https://scikit-learn.org.cn/view/662.html)(estimator, X) | 为每个输入数据点生成交叉验证的估计 |
| [`model_selection.cross_val_score`](https://scikit-learn.org.cn/view/663.html)(estimator, X) | 通过交叉验证评估分数 |
| [`model_selection.learning_curve`](https://scikit-learn.org.cn/view/667.html)(estimator, X, …) | 学习曲线。 |
| [`model_selection.permutation_test_score`](https://scikit-learn.org.cn/view/673.html)（...） | 通过排列评估交叉验证分数的重要性 |
| [`model_selection.validation_curve`](https://scikit-learn.org.cn/view/676.html)(estimator, …) | 验证曲线。 |

## sklearn.multiclass：多类和多标签分类

### 多类和多标签分类策略

该模块实现了多类学习算法：

-   一对剩余/一对全部
    
-   一对一
    
-   纠错输出代码
    

此模块中提供的估计器是元估计器：它们需要在其构造函数中提供基本估计器。例如，可以使用这些估计器将二进制分类器或回归器转换为多类分类器。也可以将这些估计器与多类估计器一起使用，以期提高其准确性或运行时性能。

scikit-learn中的所有分类器均实现多类分类；仅当您要尝试使用自定义多类别策略时，才需要使用此模块。

相对于其余的元分类器也实现了一种`predict_proba`方法，只要该方法由基本分类器实现即可。该方法在单标签和多标签情况下都返回类成员资格的概率。请注意，在多标签情况下，概率是给定样本属于给定类别的边际概率。这样，在多标签情况下，给定样本的所有可能标签上的这些概率之和_不会_像在单标签情况下那样合计为一。

**用户指南**：有关更多详细信息，请参见“[多类和多标签算法](https://scikit-learn.org.cn/view/91.html)”部分。

| [`multiclass.OneVsRestClassifier`](https://scikit-learn.org.cn/view/679.html)(estimator, \*) | 一对剩余（OvR）多类别/多标签策略 |
| --- | --- |
| [`multiclass.OneVsOneClassifier`](https://scikit-learn.org.cn/view/680.html)(estimator, \*) | 一对一多策略 |
| [`multiclass.OutputCodeClassifier`](https://scikit-learn.org.cn/view/681.html)(estimator, \*) | （错误纠正）输出代码多类策略 |

## sklearn.multioutput：多输出回归和分类

该模块实现多输出回归和分类。

此模块中提供的估计器是元估计器：它们需要在其构造函数中提供基本估计器。元估计器将单输出估计器扩展到多输出估计器。

**用户指南**：有关更多详细信息，请参见“ [多类和多标签算法”](https://scikit-learn.org.cn/view/91.html)部分。

| [`multioutput.ClassifierChain`](https://scikit-learn.org.cn/view/682.html)（base\_estimator，\*） | 将二元分类器排列到一个链中的多标签模型。 |
| --- | --- |
| [`multioutput.MultiOutputRegressor`](https://scikit-learn.org.cn/view/683.html)(estimator, \*) | 多目标回归 |
| [`multioutput.MultiOutputClassifier`](https://scikit-learn.org.cn/view/684.html)(estimator, \*) | 多目标分类 |
| [`multioutput.RegressorChain`](https://scikit-learn.org.cn/view/685.html)（base\_estimator，\*） | 一种多标签模型，可将回归安排到一个链中。 |

## sklearn.naive\_bayes：朴素贝叶斯

该[`sklearn.naive_bayes`](https://scikit-learn.org.cn/lists/3.html#sklearn.naive_bayes%EF%BC%9A%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF)模块实现了朴素贝叶斯算法。这些是基于贝叶斯定理和强（朴素）特征独立性假设的监督学习方法。

**用户指南**：有关更多详细信息，请参见“ [朴素贝叶斯”](https://scikit-learn.org.cn/view/88.html)部分。

| [`naive_bayes.BernoulliNB`](https://scikit-learn.org.cn/view/686.html)（\*\[， α， …\]） | 朴素贝叶斯分类器用于多元伯努利模型。 |
| --- | --- |
| [`naive_bayes.CategoricalNB`](https://scikit-learn.org.cn/view/687.html)（\*\[， α， …\]） | 朴素贝叶斯分类器的分类特征 |
| [`naive_bayes.ComplementNB`](https://scikit-learn.org.cn/view/688.html)（\*\[， α， …\]） | 在Rennie等人中描述的补体朴素贝叶斯分类器。 |
| [`naive_bayes.GaussianNB`](https://scikit-learn.org.cn/view/689.html)(\*\[, priors, …\]) | 高斯朴素贝叶斯（GaussianNB） |
| [`naive_bayes.MultinomialNB`](https://scikit-learn.org.cn/view/690.html)（\*\[， α， …\]） | 朴素贝叶斯分类器用于多项模型 |

## sklearn.neighbors：最近邻

该[`sklearn.neighbors`](https://scikit-learn.org.cn/lists/3.html#sklearn.neighbors%EF%BC%9A%E6%9C%80%E8%BF%91%E9%82%BB)模块实现k近邻算法。

**用户指南**：有关更多详细信息，请参见“ [最近邻](https://scikit-learn.org.cn/view/85.html)”部分。

| [`neighbors.BallTree`](https://scikit-learn.org.cn/view/691.html)（X \[，leaf\_size，metric\]） | BallTree用于快速广义N点问题 |
| --- | --- |
| [`neighbors.DistanceMetric`](https://scikit-learn.org.cn/view/692.html) | DistanceMetric类 |
| [`neighbors.KDTree`](https://scikit-learn.org.cn/view/693.html)（X \[，leaf\_size，metric\]） | KDTree用于快速广义N点问题 |
| [`neighbors.KernelDensity`](https://scikit-learn.org.cn/view/694.html)(\*\[, bandwidth, …\]) | 内核密度估计。 |
| [`neighbors.KNeighborsClassifier`](https://scikit-learn.org.cn/view/695.html)（\[…\]） | 分类器执行k最近邻居投票。 |
| [`neighbors.KNeighborsRegressor`](https://scikit-learn.org.cn/view/696.html)（\[n\_neighbors，...\]） | 基于k最近邻的回归。 |
| [`neighbors.KNeighborsTransformer`](https://scikit-learn.org.cn/view/698.html)(\*\[, mode, …\]) | 将X转换为k个最近邻居的（加权）图 |
| [`neighbors.LocalOutlierFactor`](https://scikit-learn.org.cn/view/701.html)（\[n\_neighbors，...\]） | 使用局部离群因子（LOF）的无监督离群检测 |
| [`neighbors.RadiusNeighborsClassifier`](https://scikit-learn.org.cn/view/704.html)（\[…\]） | 分类器在给定半径内实现邻居之间的投票 |
| [`neighbors.RadiusNeighborsRegressor`](https://scikit-learn.org.cn/view/705.html)(\[radius, …\]) | 基于固定半径内的邻居的回归。 |
| [`neighbors.RadiusNeighborsTransformer`](https://scikit-learn.org.cn/view/706.html)（\* \[，…\]） | 将X转换为比半径更近的邻居的（加权）图 |
| [`neighbors.NearestCentroid`](https://scikit-learn.org.cn/view/707.html)(\[metric, …\]) | 最近的质心分类器。 |
| [`neighbors.NearestNeighbors`](https://scikit-learn.org.cn/view/708.html)（\* \[，n\_neighbors，…\]） | 用于实施邻居搜索的无监督学习者。 |
| [`neighbors.NeighborhoodComponentsAnalysis`](https://scikit-learn.org.cn/view/709.html)（\[…\]） | 邻域成分分析 |
| [`neighbors.kneighbors_graph`](https://scikit-learn.org.cn/view/710.html)（X，n\_neighbors，\*） | 计算X中点的k邻居的（加权）图 |
| [`neighbors.radius_neighbors_graph`](https://scikit-learn.org.cn/view/711.html)(X, radius, \*) | 计算X中点的邻居（加权）图 |

## sklearn.neural\_network：神经网络模型

该[`sklearn.neural_network`](https://scikit-learn.org.cn/lists/3.html#sklearn.neural_network%EF%BC%9A%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B)模块包括基于神经网络的模型。

**用户指南**：有关更多详细信息，请参见[神经网络模型（受监管）](https://scikit-learn.org.cn/view/105.html)和[神经网络模型（无监管）](https://scikit-learn.org.cn/view/114.html)部分。

| [`neural_network.BernoulliRBM`](https://scikit-learn.org.cn/view/712.html)（\[n\_components，...\]） | 伯努利限制玻尔兹曼机（RBM）。 |
| --- | --- |
| [`neural_network.MLPClassifier`](https://scikit-learn.org.cn/view/713.html)（\[…\]） | 多层感知器分类器。 |
| [`neural_network.MLPRegressor`](https://scikit-learn.org.cn/view/714.html)（\[…\]） | 多层感知器回归器。 |

## sklearn.pipeline：管道

该[`sklearn.pipeline`](https://scikit-learn.org.cn/lists/3.html#sklearn.pipeline%EF%BC%9A%E7%AE%A1%E9%81%93)模块实现实用程序以构建复合估计器，作为转换和估计器链。

**用户指南**：有关更多详细信息，请参见“ [管道和复合估计器](https://scikit-learn.org.cn/view/118.html)”部分。

| [`pipeline.FeatureUnion`](https://scikit-learn.org.cn/view/715.html)（transformer\_list，\* \[，…\]） | 连接多个转换器对象的结果。 |
| --- | --- |
| [`pipeline.Pipeline`](https://scikit-learn.org.cn/view/716.html)(steps, \*\[, memory, verbose\]) | 带有最终估算器的变换管线。 |
| [`pipeline.make_pipeline`](https://scikit-learn.org.cn/view/717.html)(\*steps, \* | \*kwargs) |
| [`pipeline.make_union`](https://scikit-learn.org.cn/view/719.html)(\*transformers, \*\*kwargs) | 从给定的转换器构造一个FeatureUnion。 |

## sklearn.preprocessing：预处理和规范化

该[`sklearn.preprocessing`](https://scikit-learn.org.cn/lists/3.html#sklearn.preprocessing%EF%BC%9A%E9%A2%84%E5%A4%84%E7%90%86%E5%92%8C%E8%A7%84%E8%8C%83%E5%8C%96)模块包括缩放，居中，归一化，二值化方法。

**用户指南**：有关更多详细信息，请参见“ [预处理数据](https://scikit-learn.org.cn/view/123.html)”部分。

| [`preprocessing.Binarizer`](https://scikit-learn.org.cn/view/721.html)(\*\[, threshold, copy\]) | 根据阈值对数据进行二值化（将要素值设置为0或1） |
| --- | --- |
| [`preprocessing.FunctionTransformer`](https://scikit-learn.org.cn/view/725.html)（\[func，...\]） | 从任意可调用对象构造一个转换器。 |
| [`preprocessing.KBinsDiscretizer`](https://scikit-learn.org.cn/view/722.html)（\[n\_bins，...\]） | 将连续数据分成间隔。 |
| [`preprocessing.KernelCenterer`](https://scikit-learn.org.cn/view/724.html)（） | 将内核矩阵居中 |
| [`preprocessing.LabelBinarizer`](https://scikit-learn.org.cn/view/729.html)（\* \[，neg\_label，…\]） | 以一对一的方式对标签进行二值化 |
| [`preprocessing.LabelEncoder`](https://scikit-learn.org.cn/view/730.html) | 使用0到n\_classes-1之间的值对目标标签进行编码。 |
| [`preprocessing.MultiLabelBinarizer`](https://scikit-learn.org.cn/view/731.html)（\* \[，…\]） | 在可迭代的可迭代对象和多标签格式之间进行转换 |
| [`preprocessing.MaxAbsScaler`](https://scikit-learn.org.cn/view/732.html)(\*\[, copy\]) | 通过其最大绝对值缩放每个特征。 |
| [`preprocessing.MinMaxScaler`](https://scikit-learn.org.cn/view/733.html)(\[feature\_range, copy\]) | 通过将每个要素缩放到给定范围来变换要素。 |
| [`preprocessing.Normalizer`](https://scikit-learn.org.cn/view/736.html)(\[norm, copy\]) | 将样本分别归一化为单位范数。 |
| [`preprocessing.OneHotEncoder`](https://scikit-learn.org.cn/view/740.html)(\*\[, categories, …\]) | 将分类要素编码为一键式数字数组。 |
| [`preprocessing.OrdinalEncoder`](https://scikit-learn.org.cn/view/744.html)（\* \[，…\]） | 将分类特征编码为整数数组。 |
| [`preprocessing.PolynomialFeatures`](https://scikit-learn.org.cn/view/746.html)(\[degree, …\]) | 生成多项式和交互特征。 |
| [`preprocessing.PowerTransformer`](https://scikit-learn.org.cn/view/747.html)(\[method, …\]) | 逐个应用幂变换以使数据更像高斯型。 |
| [`preprocessing.QuantileTransformer`](https://scikit-learn.org.cn/view/748.html)（\* \[，…\]） | 使用分位数信息变换特征。 |
| [`preprocessing.RobustScaler`](https://scikit-learn.org.cn/view/751.html)（\* \[，…\]） | 使用对异常值具有鲁棒性的统计量来缩放要素。 |
| [`preprocessing.StandardScaler`](https://scikit-learn.org.cn/view/753.html)(\*\[, copy, …\]) | 通过去除均值并缩放到单位方差来标准化特征 |
| [`preprocessing.add_dummy_feature`](https://scikit-learn.org.cn/view/757.html)(X\[, value\]) | 具有附加虚拟功能的增强数据集。 |
| [`preprocessing.binarize`](https://scikit-learn.org.cn/view/758.html)(X, \*\[, threshold, copy\]) | 类数组或稀疏矩阵的布尔阈值 |
| [`preprocessing.label_binarize`](https://scikit-learn.org.cn/view/759.html)(y, \*, classes) | 以一对一的方式对标签进行二值化 |
| [`preprocessing.maxabs_scale`](https://scikit-learn.org.cn/view/760.html)(X, \*\[, axis, copy\]) | 将每个要素缩放到\[-1，1\]范围而不会破坏稀疏性。 |
| [`preprocessing.minmax_scale`](https://scikit-learn.org.cn/view/761.html)（X\[， …\]） | 通过将每个要素缩放到给定范围来变换要素。 |
| [`preprocessing.normalize`](https://scikit-learn.org.cn/view/762.html)(X\[, norm, axis, …\]) | 分别将输入向量缩放为单位范数（向量长度）。 |
| [`preprocessing.quantile_transform`](https://scikit-learn.org.cn/view/764.html)（X， \*\[， …\]） | 使用分位数信息变换特征。 |
| [`preprocessing.robust_scale`](https://scikit-learn.org.cn/view/765.html)(X, \*\[, axis, …\]) | 沿任何轴标准化数据集 |
| [`preprocessing.scale`](https://scikit-learn.org.cn/view/766.html)(X, \*\[, axis, with\_mean, …\]) | 沿任何轴标准化数据集 |
| [`preprocessing.power_transform`](https://scikit-learn.org.cn/view/768.html)(X\[, method, …\]) | 幂变换是一组参数化，单调变换，可用于使数据更像高斯型。 |

## sklearn.random\_projection：随机投影

随机投影转换器

随机投影是一种简单且计算有效的方法，可通过以可控制的精度（以附加方差）为代价来减少数据的维数，以缩短处理时间并缩小模型尺寸。

控制随机投影矩阵的尺寸和分布，以保留数据集的任何两个样本之间的成对距离。

随机投影效率背后的主要理论结果是 [Johnson-Lindenstrauss引理（引用Wikipedia）](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)：

> 在数学中，Johnson-Lindenstrauss引理是关于从高维点到低维欧几里德空间的点的低失真嵌入的结果。引理指出，高维空间中的一小部分点可以以几乎保留点之间的距离的方式嵌入到低维空间中。用于嵌入的地图至少为Lipschitz，甚至可以视为正交投影。

**用户指南**：有关更多详细信息，请参见“ [随机投影”](https://scikit-learn.org.cn/view/126.html)部分。

| [`random_projection.GaussianRandomProjection`](https://scikit-learn.org.cn/view/771.html)（\[…\]） | 通过高斯随机投影降低维数 |
| --- | --- |
| [`random_projection.SparseRandomProjection`](https://scikit-learn.org.cn/view/772.html)（\[…\]） | 通过稀疏随机投影降低尺寸 |
| [`random_projection.johnson_lindenstrauss_min_dim`](https://scikit-learn.org.cn/view/773.html)（...） | 查找“安全”数量的组件以随机投影 |

## sklearn.semi\_supervised半监督学习

该[`sklearn.semi_supervised`](https://scikit-learn.org.cn/lists/3.html#sklearn.semi_supervised%E5%8D%8A%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0)模块实现了半监督学习算法。这些算法将少量标记的数据和大量未标记的数据用于分类任务。该模块包括标签传播。

**用户指南**：有关更多详细信息，请参见“ [半监督学习](https://scikit-learn.org.cn/view/102.html)”部分。

| [`semi_supervised.LabelPropagation`](https://scikit-learn.org.cn/view/774.html)(\[kernel, …\]) | 标签传播分类器 |
| --- | --- |
| [`semi_supervised.LabelSpreading`](https://scikit-learn.org.cn/view/775.html)(\[kernel, …\]) | 用于半监督学习的LabelSpreading模型 |

## sklearn.svm：支持向量机

该[`sklearn.svm`](https://scikit-learn.org.cn/lists/3.html#sklearn.svm%EF%BC%9A%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA)模块包括支持向量机算法。

**用户指南**：有关更多详细信息，请参见“[支持向量机](https://scikit-learn.org.cn/view/83.html)”部分。

### 估计器

| [`svm.LinearSVC`](https://scikit-learn.org.cn/view/776.html)(\[penalty, loss, dual, tol, C, …\]) | 线性支持向量分类。 |
| --- | --- |
| [`svm.LinearSVR`](https://scikit-learn.org.cn/view/777.html)(\*\[, epsilon, tol, C, loss, …\]) | 线性支持向量回归。 |
| [`svm.NuSVC`](https://scikit-learn.org.cn/view/778.html)(\*\[, nu, kernel, degree, gamma, …\]) | Nu支持向量分类。 |
| [`svm.NuSVR`](https://scikit-learn.org.cn/view/779.html)(\*\[, nu, C, kernel, degree, gamma, …\]) | Nu支持向量回归。 |
| [`svm.OneClassSVM`](https://scikit-learn.org.cn/view/780.html)(\*\[, kernel, degree, gamma, …\]) | 无监督异常值检测。 |
| [`svm.SVC`](https://scikit-learn.org.cn/view/781.html)(\*\[, C, kernel, degree, gamma, …\]) | C支持向量分类。 |
| [`svm.SVR`](https://scikit-learn.org.cn/view/782.html)(\*\[, kernel, degree, gamma, coef0, …\]) | Epsilon支持向量回归。 |
| [`svm.l1_min_c`](https://scikit-learn.org.cn/view/783.html)(X, y, \*\[, loss, fit\_intercept, …\]) | 返回C的最低界限，以确保对于（l1\_min\_C，infinity）中的C，该模型不能为空。 |

## sklearn.tree：决策树

该[`sklearn.tree`](https://scikit-learn.org.cn/lists/3.html#sklearn.tree%EF%BC%9A%E5%86%B3%E7%AD%96%E6%A0%91)模块包括用于分类和回归的基于决策树的模型。

**用户指南**：有关更多详细信息，请参见“ [决策树”](https://scikit-learn.org.cn/view/89.html)部分。

| [`tree.DecisionTreeClassifier`](https://scikit-learn.org.cn/view/784.html)(\*\[, criterion, …\]) | 决策树分类器。 |
| --- | --- |
| [`tree.DecisionTreeRegressor`](https://scikit-learn.org.cn/view/785.html)(\*\[, criterion, …\]) | 决策树回归器。 |
| [`tree.ExtraTreeClassifier`](https://scikit-learn.org.cn/view/786.html)(\*\[, criterion, …\]) | 极为随机的树分类器。 |
| [`tree.ExtraTreeRegressor`](https://scikit-learn.org.cn/view/787.html)(\*\[, criterion, …\]) | 极随机的树回归器。 |
| [`tree.export_graphviz`](https://scikit-learn.org.cn/view/788.html)(decision\_tree\[, …\]) | 以DOT格式导出决策树。 |
| [`tree.export_text`](https://scikit-learn.org.cn/view/789.html)(decision\_tree, \*\[, …\]) | 建立一个文本报告，显示决策树的规则。 |

### 绘图

| [`tree.plot_tree`](https://scikit-learn.org.cn/view/790.html)（决策树， \*\[， …\]） | 绘制决策树。 |

## sklearn.utils：实用工具

该[`sklearn.utils`](https://scikit-learn.org.cn/lists/3.html#sklearn.utils%EF%BC%9A%E5%AE%9E%E7%94%A8%E5%B7%A5%E5%85%B7)模块包括各种实用程序。

| [`utils.arrayfuncs.min_pos`](https://scikit-learn.org.cn/view/791.html) | 在正值上找到数组的最小值 |
| --- | --- |
| [`utils.as_float_array`](https://scikit-learn.org.cn/view/792.html)(X, \*\[, copy, …\]) | 将类似数组的数组转换为浮点数组。 |
| [`utils.assert_all_finite`](https://scikit-learn.org.cn/view/793.html)（X，\* \[，allow\_nan\]） | 如果X包含NaN或无穷大，则引发ValueError。 |
| [`utils.Bunch`](https://scikit-learn.org.cn/view/794.html)（\*\* kwargs） | 容器对象将键公开为属性 |
| [`utils.check_X_y`](https://scikit-learn.org.cn/view/795.html)（X，y \[，accept\_sparse，…\]） | 标准估算器的输入验证。 |
| [`utils.check_array`](https://scikit-learn.org.cn/view/796.html)(array\[, accept\_sparse, …\]) | 对数组，列表，稀疏矩阵或类似内容进行输入验证。 |
| [`utils.check_scalar`](https://scikit-learn.org.cn/view/797.html)(x, name, target\_type, \*) | 验证标量参数的类型和值。 |
| [`utils.check_consistent_length`](https://scikit-learn.org.cn/view/798.html)(\*arrays) | 检查所有数组的第一维度是否一致。 |
| [`utils.check_random_state`](https://scikit-learn.org.cn/view/799.html)(seed) | 将种子转换为np.random.RandomState实例 |
| [`utils.class_weight.compute_class_weight`](https://scikit-learn.org.cn/lists/%5Bttps://scikit-learn.org.cn/view/800.html%5D(https://scikit-learn.org.cn/view/800.html))（...） | 估计不平衡数据集的类权重。 |
| [`utils.class_weight.compute_sample_weight`](https://scikit-learn.org.cn/view/801.html)（...） | 对于不平衡的数据集，按类别估算样本权重。 |
| [`utils.deprecated`](https://scikit-learn.org.cn/view/802.html)(\[extra\]) | 装饰器，用于将功能或类标记为不推荐使用。 |
| [`utils.estimator_checks.check_estimator`](https://scikit-learn.org.cn/view/803.html)(Estimator) | 检查估计器是否遵守scikit-learn约定。 |
| [`utils.estimator_checks.parametrize_with_checks`](https://scikit-learn.org.cn/view/804.html)（...） | Pytest特定的装饰器，用于参数估计器检查。 |
| [`utils.estimator_html_repr`](https://scikit-learn.org.cn/view/805.html)(estimator) | 构建估算器的HTML表示形式。 |
| [`utils.extmath.safe_sparse_dot`](https://scikit-learn.org.cn/view/806.html)（a，b，\* \[，…\]） | 正确处理稀疏矩阵案例的点积 |
| [`utils.extmath.randomized_range_finder`](https://scikit-learn.org.cn/view/807.html)(A, \*, …) | 计算一个正交矩阵，其范围近似于A的范围。 |
| [`utils.extmath.randomized_svd`](https://scikit-learn.org.cn/view/808.html)（M，n\_components，\*） | 计算截断的随机SVD |
| [`utils.extmath.fast_logdet`](https://scikit-learn.org.cn/view/809.html)(A) | 计算一个对称的log（det（A）） |
| [`utils.extmath.density`](https://scikit-learn.org.cn/view/810.html)（w，\*\* kwargs） | 计算稀疏向量的密度 |
| [`utils.extmath.weighted_mode`](https://scikit-learn.org.cn/view/811.html)(a, w, \*\[, axis\]) | 返回数组中加权模态（最常见）值的数组 |
| [`utils.gen_even_slices`](https://scikit-learn.org.cn/view/812.html)（n，n\_packs，\* \[，n\_samples\]） | 生成器创建n\_packs片，最多可达n。 |
| [`utils.graph.single_source_shortest_path_length`](https://scikit-learn.org.cn/view/813.html)（...） | 返回从源到所有可达节点的最短路径长度。 |
| [`utils.graph_shortest_path.graph_shortest_path`](https://scikit-learn.org.cn/view/814.html) | 对正有向图或无向图执行最短路径图搜索。 |
| [`utils.indexable`](https://scikit-learn.org.cn/view/815.html)(\*iterables) | 使数组可索引以进行交叉验证。 |
| [`utils.metaestimators.if_delegate_has_method`](https://scikit-learn.org.cn/view/816.html)（...） | 为委托给子估计器的方法创建一个装饰器 |
| [`utils.multiclass.type_of_target`](https://scikit-learn.org.cn/view/817.html)（y） | 确定目标指示的数据类型。 |
| [`utils.multiclass.is_multilabel`](https://scikit-learn.org.cn/view/818.html)（y） | 检查是否`y`为多标签格式。 |
| [`utils.multiclass.unique_labels`](https://scikit-learn.org.cn/view/819.html)(\*ys) | 提取唯一标签的有序数组 |
| [`utils.murmurhash3_32`](https://scikit-learn.org.cn/view/820.html) | 计算种子的密钥的32位murmurhash3。 |
| [`utils.resample`](https://scikit-learn.org.cn/view/821.html)(\*arrays, \*\*options) | 以一致的方式对数组或稀疏矩阵重新采样 |
| [`utils._safe_indexing`](https://scikit-learn.org.cn/view/822.html)(X, indices, \*\[, axis\]) | 使用索引返回X的行，项目或列。 |
| [`utils.safe_mask`](https://scikit-learn.org.cn/view/823.html)(X, mask) | 返回可在X上安全使用的口罩。 |
| [`utils.safe_sqr`](https://scikit-learn.org.cn/view/824.html)(X, \*\[, copy\]) | 类数组和稀疏矩阵的元素明智平方。 |
| [`utils.shuffle`](https://scikit-learn.org.cn/view/825.html)(\*arrays, \*\*options) | 以一致的方式随机排列数组或稀疏矩阵 |
| [`utils.sparsefuncs.incr_mean_variance_axis`](https://scikit-learn.org.cn/view/826.html)（X， …） | 计算CSR或CSC矩阵上沿轴的增量平均值和方差。 |
| [`utils.sparsefuncs.inplace_column_scale`](https://scikit-learn.org.cn/view/827.html)(X, scale) | CSC / CSR矩阵的就地列缩放。 |
| [`utils.sparsefuncs.inplace_row_scale`](https://scikit-learn.org.cn/view/828.html)(X, scale) | CSR或CSC矩阵的就地行缩放。 |
| [`utils.sparsefuncs.inplace_swap_row`](https://scikit-learn.org.cn/view/829.html)（X，m，n） | 就地交换两行CSC / CSR矩阵。 |
| [`utils.sparsefuncs.inplace_swap_column`](https://scikit-learn.org.cn/view/830.html)（X，m，n） | 就地交换两列CSC / CSR矩阵。 |
| [`utils.sparsefuncs.mean_variance_axis`](https://scikit-learn.org.cn/view/831.html)(X, axis) | 计算CSR或CSC矩阵上沿轴的均值和方差 |
| [`utils.sparsefuncs.inplace_csr_column_scale`](https://scikit-learn.org.cn/view/832.html)（X， …） | CSR矩阵的就地列缩放。 |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l1`](https://scikit-learn.org.cn/view/833.html) | 使用l1范数进行就地行归一化 |
| [`utils.sparsefuncs_fast.inplace_csr_row_normalize_l2`](https://scikit-learn.org.cn/view/834.html) | 使用l2范数进行就地行归一化 |
| [`utils.random.sample_without_replacement`](https://scikit-learn.org.cn/view/835.html) | 采样整数而不进行替换。 |
| [`utils.validation.check_is_fitted`](https://scikit-learn.org.cn/view/836.html)(estimator) | 对估算器执行is\_fitted验证。 |
| [`utils.validation.check_memory`](https://scikit-learn.org.cn/view/837.html)(memory) | 检查`memory`是否类似于joblib.Memory。 |
| [`utils.validation.check_symmetric`](https://scikit-learn.org.cn/view/838.html)(array, \*\[, …\]) | 确保该数组是2D，正方形和对称的。 |
| [`utils.validation.column_or_1d`](https://scikit-learn.org.cn/view/839.html)(y, \*\[, warn\]) | Ravel列或一维numpy数组，否则引发错误 |
| [`utils.validation.has_fit_parameter`](https://scikit-learn.org.cn/view/840.html)（...） | 检查估计器的fit方法是否支持给定参数。 |
| [`utils.all_estimators`](https://scikit-learn.org.cn/view/841.html)（\[type\_filter\]） | 从sklearn获取所有估计量的列表。 |

来自joblib的实用程序：

| [`utils.parallel_backend`](https://scikit-learn.org.cn/view/842.html)(backend\[, n\_jobs, …\]) | 在with块中更改Parallel使用的默认后端。 |
| --- | --- |
| [`utils.register_parallel_backend`](https://scikit-learn.org.cn/view/843.html)(name, factory) | 注册一个新的并行后端工厂。 |

## 最近不推荐使用的

### 在0.24中删除

| [`model_selection.fit_grid_point`](https://scikit-learn.org.cn/view/844.html)（X，y，…\[，…\]） | 不推荐使用：fit\_grid\_point在0.23版中已弃用，并将在0.25版中删除 |
| --- | --- |
| [`utils.safe_indexing`](https://scikit-learn.org.cn/view/845.html)(X, indices, \*\[, axis\]) | 不推荐使用：safe\_indexing在0.22版中已弃用，并将在0.24版中删除。 |