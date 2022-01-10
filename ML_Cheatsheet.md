# ML Cheatsheet
dl
http://jalammar.github.io/illustrated-transformer/
http://jalammar.github.io/illustrated-transformer/�https://tech.meituan.com/2019/11/14/nlp-bert-practice.html


https://huggingface.co/models

## ML Models
| Name                | Method                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | trade off                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear Regression   | Least of square<br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586360907726_image.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | ***线性回归优点：***<br>　　实现简单，计算简单；<br>　　***缺点：***<br>　　不能拟合非线性数据；                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Logistic Regression | 这个最好从广义线性模型的角度分析，逻辑回归是假设y服从Bernoulli分布。<br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586360851511_image.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | ***Logistic回归优点：***<br>　　1、实现简单；<br>　　2、分类时计算量非常小，速度很快，存储资源低；<br>　　***缺点：***<br>　　1、容易欠拟合，一般准确度不太高<br>　　2、只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Naive Bayes         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| SVM                 | 间隔最大化来得到最优分离超平面。方法是将这个问题形式化为一个凸二次规划问题，还可以等价位一个正则化的合页损失最小化问题。SVM又有硬间隔最大化和软间隔SVM两种。这时首先要考虑的是如何定义间隔，这就引出了函数间隔和几何间隔的概念（这里只说思路），我们选择了几何间隔作为距离评定标准（**为什么要这样，怎么求出来的要知道**），我们希望能够最大化与超平面之间的几何间隔x，同时要求所有点都大于这个值，通过一些变化就得到了我们常见的SVM表达式。接着我们发现定义出的x只是由个别几个支持向量决定的。对于原始问题（primal problem）而言，可以利用凸函数的函数包来进行求解，但是发现如果用对偶问题（dual ）求解会变得更简单，而且可以引入核函数。而原始问题转为对偶问题需要满足KKT条件（这个条件应该细细思考一下）到这里还都是比较好求解的。因为我们前面说过可以变成软间隔问题，引入了惩罚系数，这样还可以引出hinge损失的等价形式（这样可以用梯度下降的思想求解SVM了）。我个人认为难的地方在于求解参数的SMO算法。<br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586361168972_image.png)<br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586361191334_image.png) | ***SVM算法优点：***<br>　　可用于线性/非线性分类，也可以用于回归；<br>　　低泛化误差；<br>　　容易解释；<br>　　计算复杂度较低；<br>　　***缺点：***<br>　　对参数和核函数的选择比较敏感；<br>　　原始的SVM只比较擅长处理二分类问题；<br><br><br>LR和SVM最大的区别在于损失函数的选择，LR的损失函数为Log损失（或者说是逻辑损失都可以）、而SVM的损失函数为hinge loss。 <br><br>- 最后，SVM只考虑支持向量（也就是和分类相关的少数点）<br><br><br>**（1）带核的SVM为什么能分类非线性问题？** <br>核函数的本质是两个函数的內积，而这个函数在SVM中可以表示成对于输入值的高维映射。注意核并不是直接对应映射，核只不过是一个內积<br><br>**（2）RBF核一定是线性可分的吗** <br>不一定，RBF核比较难调参而且容易出现维度灾难，要知道无穷维的概念是从泰勒展开得出的。  <br><br>**（3）常用核函数及核函数的条件：** <br>核函数选择的时候应该从线性核开始，而且在特征很多的情况下没有必要选择高斯核，应该从简单到难的选择模型。我们通常说的核函数指的是正定和函数，其充要条件是对于任意的x属于X，要求K对应的Gram矩阵要是半正定矩阵。<br><br>- RBF核径向基，这类函数取值依赖于特定点间的距离，所以拉普拉斯核其实也是径向基核。<br>- 线性核：主要用于线性可分的情况<br>- 多项式核<br><br>**（6）处理数据偏斜**<br>可以对数量多的类使得惩罚系数C越小表示越不重视，相反另数量少的类惩罚系数变大。 |
| KNN                 | KNN即最近邻算法，其主要过程为：<br>　　1. 计算训练样本和测试样本中每个样本点的距离（常见的距离度量有欧式距离，马氏距离等）；<br>　　2. 对上面所有的距离值进行排序；<br>　　3. 选前k个最小距离的样本；<br>　　4. 根据这k个样本的标签进行投票，得到最后的分类类别；<br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586360981207_image.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | ***KNN算法的优点：***<br>　　1. 思想简单，理论成熟，既可以用来做分类也可以用来做回归；<br>　　2. 可用于非线性分类；<br>　　3. 训练时间复杂度为O(n)；<br>　　4. 准确度高，对数据没有假设，对outlier不敏感；<br>　　***缺点：***<br>　　1. 计算量大；<br>　　2. 样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；<br>　　3. 需要大量的内存；<br><br><br>　　如何选择一个最佳的K值，这取决于数据。一般情况下，在分类时较大的K值能够减小噪声的影响。但会使类别之间的界限变得模糊                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| K-means             | K-means, k-medoids(每一个类别中找一个样本点来代表),CLARANS.<br>　　k-means是使下面的表达式值最小：<br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586361288446_image.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | ***k-means算法的优点：***<br>　　（1）k-means算法是解决聚类问题的一种经典算法，算法简单、快速。<br>　　（2）对处理大数据集，该算法是相对可伸缩的和高效率的，因为它的复杂度大约是O(nkt)，其中n是所有对象的数目，k是簇的数目,t是迭代的次数。通常k<<n。这个算法通常局部收敛。<br>　　（3）算法尝试找出使平方误差函数值最小的k个划分。当簇是密集的、球状或团状的，且簇与簇之间区别明显时，聚类效果较好。<br>　　 ***缺点：***<br>　　（1）k-平均方法只有在簇的平均值被定义的情况下才能使用，且对有些分类属性的数据不适合。<br>　　（2）要求用户必须事先给出要生成的簇的数目k。<br>　　（3）对初值敏感，对于不同的初始值，可能会导致不同的聚类结果。<br>　　（4）不适合于发现非凸面形状的簇，或者大小差别很大的簇。<br>　　（5）对于"噪声"和孤立点数据敏感，少量的该类数据能够对平均值产生极大影响。                                                                                                                                                                                                                                                                                                        |
| Decision Tree       | **1、决策树树相关问题**<br><br>- **（1）各种熵的计算** <br>- 熵、联合熵、条件熵、交叉熵、KL散度（相对熵）<br>- 熵用于衡量不确定性，所以均分的时候熵最大<br>- KL散度用于度量两个分布的不相似性，KL(p||q)等于交叉熵H(p,q)-熵H(p)。交叉熵可以看成是用q编码P所需的bit数，减去p本身需要的bit数，KL散度相当于用q编码p需要的额外bits。<br>- 交互信息Mutual information ：I(x,y) = H(x)-H(x|y) = H(y)-H(y|x) 表示观察到x后，y的熵会减少多少。<br><br>**（2）常用的树搭建方法：ID3、C4.5、CART** <br>上述几种树分别利用信息增益、信息增益率、Gini指数作为数据分割标准。其中信息增益衡量按照某个特征分割前后熵的减少程度，其实就是上面说的交互信息。 用上述信息增益会出现优先选择具有较多属性的特征，毕竟分的越细的属性确定性越高。所以提出了信息增益率的概念，让含有较多属性的特征的作用降低。 CART树在**分类过程**中使用的基尼指数Gini，只能用于切分二叉树，而且和ID3、C4.5树不同，Cart树不会在每一个步骤删除所用特征。 <br><br>**（3）防止过拟合：剪枝** <br>剪枝分为前剪枝和后剪枝，前剪枝本质就是早停止，后剪枝通常是通过衡量剪枝后损失函数变化来决定是否剪枝。后剪枝有：错误率降低剪枝、悲观剪枝、代价复杂度剪枝<br><br>**（4）前剪枝的停止条件**<br><br>- 节点中样本为同一类<br>- 特征不足返回多类<br>- 如果某个分支没有值则返回父节点中的多类<br>- 样本个数小于阈值返回多类      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Random Forecast     | 随机森林改变了决策树容易过拟合的问题，这主要是由两个操作所优化的：1、Boostrap从袋内有放回的抽取样本值2、每次随机抽取一定数量的特征（通常为sqr(n)）。 <br><br>- 分类问题：采用Bagging投票的方式选择类别频次最高的 <br>- 回归问题：直接取每颗树结果的平均值。<br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586200675421_image.png)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 随机森林等树算法都是非线性的，而LR是线性的。LR更侧重全局优化，而树模型主要是局部的优化。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Boosted Tree (GBT)  | GBDT(Gradient Boosting Decision Tree) 又叫 MART（Multiple Additive Regression Tree)，好像在阿里内部用得比较多（所以阿里算法岗位面试时可能会问到），它是一种迭代的决策树算法，该算法由多棵决策树组成，所有树的输出结果累加起来就是最终答案。它在被提出之初就和SVM一起被认为是泛化能力（generalization)较强的算法。近些年更因为被用于搜索排序的机器学习模型而引起大家关注。<br>　　GBDT是回归树，不是分类树。其核心就在于，每一棵树是从之前所有树的残差中来学习的。为了防止过拟合，和Adaboosting一样，也加入了boosting这一项。<br><br><br>**（3）Boosting之GBDT** <br>将基分类器变成二叉树，回归用二叉回归树，分类用二叉分类树。和上面的Adaboost相比，回归树的损失函数为平方损失，同样可以用指数损失函数定义分类问题。但是对于一般损失函数怎么计算呢？GBDT（梯度提升决策树）是为了解决一般损失函数的优化问题，方法是用损失函数的负梯度在当前模型的值来模拟回归问题中残差的近似值。 （**注：**由于GBDT很容易出现过拟合的问题，所以推荐的GBDT深度不要超过6，而随机森林可以在15以上。）                                                                                                                                                                                       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Xgboost             | 这个工具主要有以下几个特点：<br><br>- 支持线性分类器<br>- 可以自定义损失函数，并且可以用二阶偏导<br>- 加入了正则化项：叶节点数、每个叶节点输出score的L2-norm<br>- 支持特征抽样<br>- 在一定情况下支持并行，只有在建树的阶段才会用到，每个节点可以并行的寻找分裂特征。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Light BT            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## DL Models
## Loss Functions
- **Loss**: Used to evaluate and diagnose model optimization only.
- **Metric**: Used to evaluate and choose models in the context of the project.
## [**Cross-Entropy**](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#id11) **/ log loss**

Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label.
An important aspect of this is that cross entropy loss penalizes heavily the predictions that are *confident but wrong*.

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1585756190041_image.png)

    def CrossEntropy(yHat, y):
        if y == 1:
          return -log(yHat)
        else:
          return -log(1 - yHat)

https://gombru.github.io/2018/05/23/cross_entropy_loss/

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1587443595815_image.png)

## [**Hinge**](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#id12)

***Hinge loss is primarily used with*** [***Support Vector Machine (SVM)***](https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/) ***Classifiers with class labels -1 and 1. (alternative to  cross-entropy)***
The hinge loss function encourages examples to have the correct sign, assigning more error when there is a difference in the sign between the actual and predicted class values.
Reports of performance with the hinge loss are mixed, sometimes resulting in better performance than cross-entropy on binary classification problems.

$$L = max(0, 1-\hat{y} * y)$$


    def Hinge(yHat, y):
        return np.max(0, 1 - yHat * y)

in linear SVMs, $$y=w x + b$$
when $$\hat{y}$$ and $$y$$ have the same sign (meaning $$y$$ predicting the right class) and $$|y|\geq 1$$ the hinge loss is 0.  When they have opposite signs, loss increases linearly with $$y$$, and similiarly if $$|y| \lt 1$$ even if it has the same sign( correct prediction, but not by enough margin)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586359959761_image.png)

## [**Huber**](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#id13)
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1585756400996_image.png)
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1585756655610_image.png)


Typically used for regression. It’s less sensitive to outliers than the MSE as it treats error as square only inside an interval.


![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1585756658620_image.png)



    def Huber(yHat, y, delta=1.):
        return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))
## [**MAE (L1)**](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#id15)

Mean Absolute Error, or L1 loss.
On some regression problems, the distribution of the target variable may be mostly Gaussian, but may have outliers, e.g. large or small values far from the mean value.
The Mean Absolute Error, or MAE, loss is an appropriate loss function in this case as it is more robust to outliers

$$MAE = \frac{\sum_i \hat{y_i} - y_i}{n}$$

    def L1(yHat, y):
        return np.sum(np.absolute(yHat - y)) / y.size
## [**MSE (L2)**](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#id16)

Mean Squared Error, or L2 loss
$$MSE = \frac{\sum_i (\hat{y_i} - y_i)^2}{n}$$

    def MSE(yHat, y):
        return np.sum((yHat - y)**2) / y.size
## References
- https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html



https://startupsventurecapital.com/essential-cheat-sheets-for-machine-learning-and-deep-learning-researchers-efb6a8ebd2e5

https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/




## Interview

**PR Curve vs ROC Curve**

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586300827022_image.png)

    - 
- **Precision** = $$TP / (TP + FP)$$
- **Recall** = $$TP / (TP+FN)$$   →  ( true positive rate, sensitivity)


    - **TruePositiveRate** = $$TP/ (TP + FN)$$
    - **FalsePositiveRate**  = $$FP/(FP+TN)$$


- $$F_1 = 2* (precision * recall )/ (precision + recall)$$
|                                                                            | ROC Curve                                                                                                                                                                                                                                                                                                                                                                             | PR Curve                                                                                                                                                                                      |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                                                            | Plot of False Positive Rate (x) vs. True Positive Rate (y).<br><br>- It is a popular diagnostic tool for classifiers on balanced and imbalanced binary prediction problems alike because it is not biased to the majority or minority class.<br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586300918526_image.png) | Plot of Recall (x) vs Precision (y).<br><br><br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586301084203_image.png) |
|                                                                            | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586300482260_image.png)                                                                                                                                                                                                                                                 | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586300474940_image.png)                                                         |
| AUC                                                                        | *AUCROC can be interpreted as the probability that the scores given by a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one.*                                                                                                                                                                                                        | The score can then be used as a point of comparison between different models on a binary classification problem where a score of 1.0 represents a model with perfect skill.                   |
|                                                                            | - ROC curves should be used when there are roughly equal numbers of observations for each class.                                                                                                                                                                                                                                                                                      | - Precision-Recall curves should be used when there is a moderate to large class imbalance.                                                                                                   |
| balance:<br><br>- 80 +<br>- 80 -<br><br>Imbalanced<br><br>- 80+<br>- 720 - | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586302735345_image.png)                                                                                                                                                                                                                                                 | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586302740197_image.png)                                                         |



**L1 and L2 regularization** 

- Calculate the sum of the absolute values of the weights, called L1.
- Calculate the sum of the squared values of the weights, called L2.

biggest reasons for regularization are 1) to avoid overfitting by not generating high coefficients for predictors that are sparse.   2) to stabilize the estimates especially when there's collinearity in the data.  

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586193349132_image.png)


The implication of this is that the L1 regularization gives you **sparse** estimates (shrinks less important features’ coefficient to zero → feature selection)

- weight regularization to a neural network has the effect of reducing generalization error and of allowing the model to pay less attention to less relevant input variables.

- can be used in DNN with a regularization term in the cost function
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586199268608_image.png)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586199276173_image.png)


When choosing the regularization term **α.** The goal is to strike the right balance between low complexity of the model and accuracy

- If your alpha value is too high, your model will be simple, but you run the risk of *underfitting* your data. Your model won’t learn enough about the training data to make useful predictions.
- If your alpha value is too low, your model will be more complex, and you run the risk of *overfitting* your data. Your model will learn too much about the particularities of the training data, and won’t be able to generalize to new data.

**Bias Variance Trade Off**
If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data.

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586193127660_image.png)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586193071960_image.png)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586193254772_image.png)


**ROC**

深度学习模型参数初始化都有哪些方法？
（1）Gaussian 满足mean=0，std=1的高斯分布x∼N(mean，std2)
（2）Xavier 满足x∼U(−a,+a)x∼U(−a,+a)的均匀分布， 其中 a = sqrt(3/n)
（3）MSRA 满足x∼N(0,σ2)x∼N(0,σ2)的高斯分布，其中σ = sqrt(2/n)
（4）Uniform 满足min=0,max=1的均匀分布。x∼U(min,max)x∼U(min,max)
等等 

如何提高小型网络的精度？
（1）模型蒸馏技术(https://arxiv.org/abs/1503.02531)
（2）利用AutoML进行网络结构的优化，可将网络计算复杂度作为约束条件之一，得到更优的结构。(https://arxiv.org/abs/1807.11626)


什么是神经网络的梯度消失问题，为什么会有梯度消失问题？有什么办法能缓解梯度消失问题？
在反向传播算法计算每一层的误差项的时候，需要乘以本层激活函数的导数值，如果导数值接近于0，则多次乘积之后误差项会趋向于0，而参数的梯度值通过误差项计算，这会导致参数的梯度值接近于0，无法用梯度下降法来有效的更新参数的值。

改进激活函数，选用更不容易饱和的函数，如ReLU函数。


1x1卷积有什么用途？
通道降维，保证卷积神经网络可以接受任何尺寸的输入数据


## Gradient Descent

Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize neural networks. 

|                                 |                                                                                                                                       |                                                                                                                                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Batch Gradient Descent          | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586371767121_image.png) | for i in range(epochs):<br>  params_grad = evaluate_gradient(loss_function, data, params)<br>  params = params - learning_rate * params_grad                                                                                          |
| **Stochastic gradient descent** | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586371814566_image.png) | for i in range(epochs):<br>  np.random.shuffle(data)<br>  for example in data:<br>    params_grad = evaluate_gradient(loss_function, example, params)<br>    params = params - learning_rate * params_grad                            |
| **Mini-batch gradient descent** | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586371902828_image.png) | for i in range(nb_epochs):<br>  np.random.shuffle(data)<br>  for batch in get_batches(data, batch_size=50):<br>    params_grad = evaluate_gradient(loss_function, batch, params)<br>    params = params - learning_rate * params_grad |
|                                 |                                                                                                                                       |                                                                                                                                                                                                                                       |
|                                 |                                                                                                                                       |                                                                                                                                                                                                                                       |



## Optimizer 




|                                     |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Momentum                            | Momentum [[5]](https://ruder.io/optimizing-gradient-descent/index.html#fn5) is a method that helps accelerate SGD in the relevant direction and dampens oscillations                                                                                                                                                                   | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586372404860_image.png)                                                                                                                                                          |
| # **Nesterov accelerated gradient** |                                                                                                                                                                                                                                                                                                                                        | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586372564753_image.png)                                                                                                                                                          |
| ## **Adagrad**                      | t adapts the learning rate to the parameters, performing smaller updates<br>(i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data. |                                                                                                                                                                                                                                                                                                |
| ## **Adadelta**                     | Adadelta [[13]](https://ruder.io/optimizing-gradient-descent/index.html#fn13) is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size <br>w         |                                                                                                                                                                                                                                                                                                |
| ## **RMSprop**                      |                                                                                                                                                                                                                                                                                                                                        | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586375169394_image.png)                                                                                                                                                          |
| ## **Adam**                         | Adam can be viewed as a combination of RMSprop and momentum: RMSprop contributes the exponentially decaying average of past squared gradients vt, while momentum accounts for the exponentially decaying average of past gradients mt.                                                                                                 | ![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586375241935_image.png)<br><br><br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586375256079_image.png) |
| ## **Nadam**                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                |
| ## **AMSGrad**                      |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                |
|                                     |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                |



**解决overfitting的方法** 
dropout， regularization， batch normalization，early stop, 但是要注意dropout只在训练的时候用，让一部分神经元随机失活。 Batch normalization是为了让输出都是单位高斯激活，方法是在连接和激活函数之间加入BatchNorm层，计算每个特征的均值和方差进行规则化。 


**model recalibration (after negative down sampling)**
Negative downsampling can speed up training and improve model performance.  However, the CTR is calibrated. For example, if the average CTR before sampling is 0.1% and we do a 0.01 negative downsampling, the empirical CTR will become roughly 10%.
$$q = \frac{p}{p+(1-p)/w}$$
where $$p$$ is prediction in downsampling space and $$w$$ is the negative downsampling rate

**Embedding**
 [In the context of neural networks, embeddings](https://www.tensorflow.org/guide/embedding) are *low-dimensional,* *learned* continuous vector representations of high-dimension sparse/dense data. 

- word2vec
    - Word2vec relies on the **distributional hypothesis** to map semantically similar words to geometrically close embedding vectors.
- learn embedding as part of DNN
- dimension = 4th root of possible values

Embedding as a Tool

- embedding map items (e.g. movies, text, users,…) to low-dimensional real vectors in a way that similar items are close to each other
- embedding can also be applide to dense data (e.g. image, audio) to create meaningful similarity metric
- jointly embedding diverse data types (e.g. text, images, audio, …) defines a similarity between them


1. **在深度学习网络中作为Embedding层，完成从高维稀疏特征向量到低维稠密特征向量的转换；**
2. **作为预训练的Embedding特征向量，与其他特征向量连接后一同输入深度学习网络进行训练；**
3. **通过计算用户和物品的Embedding相似度，Embedding可以直接作为推荐系统或计算广告系统的召回层或者召回方法之一。**

 

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586314074240_image.png)

## Recommendation System

candidate generation → scoring → re-ranking

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586315934978_image.png)

|               | content based filtering                                                                                                                   | collaborative filtering                                                                                                                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| definitions   | uses similarity between items to recommend items similar to what the user likes                                                           | uses the similarities between queries and items simultaneously to provide recommendation                                                                                                                                                    |
| example       | if user watches two cute cat videos, then the system can recommend cute animal videos to the user                                         | if user A is similar to user B, and user B likes video 1, then the system can recommend video 1 to user A (even if user A hasn’t seen any videos similar to video 1                                                                         |
| similarity    | Using Dot Product as a Similarity Measure  $$<x,y> = \sum_i x_i y_i$$                                                                     |                                                                                                                                                                                                                                             |
| methods       |                                                                                                                                           | - matrix factorization<br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586316621775_image.png)                                                                             |
| advantages    | - the model doesn’t need data from other users → scale<br>- the model can capture specific interests of a user → recommend niche contents | - no domain knowledge<br>- model can help users discover new interests<br>- matrix factorization → no need of contextual features → can be used in candidate generation                                                                     |
| disadvantages | - domain knowledge + hand-engineered features<br>- only recommend based on existing interests → hard to expand                            | - cannot handle fresh items → cold start<br>- hard to include side features for query/item<br><br><br>![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586316865832_image.png) |

A similarity measure is a function  that takes a pair of embeddings and returns a scalar measuring their similarity. 

- cosine
- dot product
- Euclidean distance

 
**Recommendation with DNN**

- youtube
    - training: weight logistic regression
    - serving: $$e^{Wx+b}$$
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586317581854_image.png)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586317672585_image.png)



- two tower
    - 现在业界的很多公司其实采用了“复杂网络离线训练，生成 embedding 存入内存数据库，线上实现 LR 或浅层 NN 等轻量级模型拟合优化目标”的上线方式。百度曾经成功应用的“双塔”模型是非常典型的例子
    - 百度的双塔模型分别用复杂网络对“用户特征”和“广告特征”进行了 embedding 化，在最后的交叉层之前，用户特征和广告特征之间没有任何交互，这就形成了两个独立的“塔”，因此称为双塔模型。
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586317340938_image.png)

    - 
    

 

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586317007858_image.png)

- Matrix factorization is usually the better choice for large corpora. It is easier to scale, cheaper to query, and less prone to folding.
- DNN models can better capture personalized preferences, but are harder to train and more expensive to query. DNN models are preferable to matrix factorization for scoring because DNN models can use more features to better capture relevance. Also, it is usually acceptable for DNN models to fold, since you mostly care about ranking a pre-filtered set of candidates assumed to be relevant.
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586317188515_image.png)


Spark ML
**Spark的并行梯度下降方法是同步阻断式的，且模型参数需通过全局广播的形式发送到各节点**，因此Spark的并行梯度下降是相对低效的。


## Online Serving
- **自研平台**
- **预训练embedding+线上轻量级模型**
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586291209406_image.png)

    - 在完成双塔模型的训练后，可以把最终的用户embedding和广告embedding存入内存数据库。而在线上inference时，也不用复现复杂网络，只需要实现最后一层的逻辑，在从内存数据库中取出用户embedding和广告embedding之后，通过简单计算即可得到最终的预估结果。




- **PMML （**Predictive Model Markup Language）
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586291277134_image.png)

- **TensorFlow Serving等原生serving平台**
    - TensorFlow Serving最普遍也是最便捷的serving方式是使用Docker建立模型Serving API
    - H2O
    


## Distributed Training

**paremeter server**

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586290443195_image.png)


Parameter Server的主要目的就是分布式并行进行梯度下降的计算完成参数的更新与最终收敛. 由于公式中正则化项的存在需要汇总所有模型参数才能够正确计算，因此较难进行模型参数的并行训练，因此Parameter Server采取了和Spark MLlib一样的**数据并行训练产生局部梯度，再汇总梯度更新参数权重**的并行化训练


1. **用异步非阻断式的分布式梯度下降策略替代同步阻断式的梯度下降策略；**
2. **实现多server节点的架构，避免了单master节点带来的带宽瓶颈和内存瓶颈；**
3. **使用一致性哈希，range pull和range push等工程手段实现信息的最小传递，避免广播操作带来的全局性网络阻塞和带宽浪费。**
![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586290397773_image.png)




**XGboost 面试超级套餐**

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586188810792_image.png)



![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586188873547_image.png)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586188896538_image.png)

![](https://paper-attachments.dropbox.com/s_D634C6852F5C3939E31EC0D32C187C73B4F8FDA6E193CC1C9D9F2A43A09E84D5_1586188913592_image.png)


梯度下降是用平面来逼近局部，牛顿法是用曲面逼近局部等等。

针对梯度爆炸问题，解决方案是引入Gradient Clipping(梯度裁剪)。通过Gradient Clipping，将梯度约束在一个范围内，这样不会使得梯度过大。


- softmax函数的定义是什么？（**知识**）
- 神经网络为什么会产生梯度消失现象？（**知识**）
- 常见的激活函数有哪些？都有什么特点？（**知识**）
- 挑一种激活函数推导梯度下降的过程。（**知识+逻辑**）
- Attention机制什么？（**知识**）
- 阿里是如何将attention机制引入推荐模型的？（**知识+业务**）
- DIN是基于什么业务逻辑引入attention机制的？（**业务**）
- DIN中将用户和商品进行了embedding，请讲清楚两项你知道的embedding方法。（**知识**）
- 你如何serving类似DIN这样的深度学习模型(**工具+业务**)
# References

- https://www.zhihu.com/question/62482926
- https://www.deeplearning-academy.com/p/ai-wiki-regularization
- https://mp.weixin.qq.com/s?__biz=MzIxODM4MjA5MA==&mid=2247486331&idx=1&sn=abc69ee44d932dd7d6bc4bbef82045c8&chksm=97ea211ea09da808a546c4a6f485f45289fea2e69cddc95aaf16f06ddfb9c428ecf9d259c980&scene=0&key=f7bb43d4492422e0472e06f4faf0076f9d9de975e8a73050e15cd63f1f549a060f9018009aa9f1f5f19aa37f1408ecb3ea2be5b8464b4eae89884e1d881c91ebef20c84ea9198fed470f36016f54c30a&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=fLkIwHFJHv2/bP8RGzzYjXOZFDZBSwG0jwcmIEQhVi6CQoLDsa0PiCF8xoyKnPtI


- http://josh-tobin.com/assets/pdf/troubleshooting-deep-neural-networks-01-19.pdf
- https://towardsdatascience.com/architecting-a-machine-learning-pipeline-a847f094d1c7
- https://acutecaretesting.org/en/articles/precision-recall-curves-what-are-they-and-how-are-they-used

- https://developers.google.com/machine-learning/recommendation/collaborative/matrix

- https://www.cnblogs.com/tornadomeet/p/3395593.html
- http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- https://ruder.io/optimizing-gradient-descent/index.html#adam
https://towardsdatascience.com/a-complete-guide-to-principal-component-analysis-pca-in-machine-learning-664f34fc3e5a









https://online.stat.psu.edu/stat800/node/1/�https://www.1point3acres.com/bbs/thread-490321-1-1.html�https://www.quora.com/q/agdkymlytzlptely�https://labs.pinterest.com/user/themes/pin_labs/assets/paper/p2p-www17.pdf�

