# 多孔材料薄覆盖层扩展反应边界建模用于时域房间声学模拟

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/61067af0e10238ed13ffe85864db96029c4beddd791f2d1d0ccc21153ff064a2.jpg)

王惠清<sup>*</sup>，Maarten Hornikx 

建筑声学，建筑环境系，埃因霍温科技大学，P.O. Box 513, 5600 MB 埃因霍温, 荷兰 

文章信息 

关键词： 时域室内声学建模 多孔层的扩展反应 薄覆盖材料 高阶精度 精确黎曼求解器 

## 摘要

声学边界条件建模对房间声学模拟的准确性有重大影响，在室内建筑环境的设计阶段以提高声学舒适度方面发挥着重要作用。在这项工作中，提出了一种基于不连续伽辽金（DG）方法的数值框架，用于对薄材料覆盖的多孔吸收体的扩展反应边界进行建模。通过将多孔材料视为子域来应用域分解方法。等效流体模型用于描述多孔材料的声学特性，其有效密度和压缩率作为无理函数由频域中的多极有理函数近似。通过采用辅助微分方程方法计算时间卷积，多孔材料的增广时域控制方程可以用与线性声学方程相同的统一双曲形式表示，这进一步实现了整个域内一致的迎风数值通量公式。通过解决基本的黎曼问题来处理传播介质之间界面上的数值耦合。与用于室内声学扩展反应边界建模的现有 DG 方法相比，导出的迎风数值通量公式不涉及辅助变量的计算。所提出的框架产生了一个适定的线性双曲系统，其边界条件受“均匀 Kreiss 条件”（Kreiss，1970）的指导。通过考虑柔软渗透膜模型来说明覆盖材料的声学特性。利用局部时间步进方法来提高计算效率。针对一维解析解进行数值验证，以验证所需的高阶收敛速度。球面波前建模的 3D 案例研究证明了该公式的宽带精度。 

## 1.简介

由于多孔材料具有宽带吸声能力和成本效益，因此对于各种声学应用至关重要。例如，它们可以作为边界表面的必要声学处理，以提高建筑环境的声学舒适度。因此，对多孔吸声体附近声波传播的精确建模一直是正在进行的研究的主题[1]。在室内声学建模中，多孔材料的吸声特性通常通过几何声学方法 [2] 的吸收系数或基于波的方法 [3] 的表面阻抗来表征。这些特性从根本上取决于波的频率和入射角，表现出所谓的扩展反应 (ER) 行为，这对严格的数值建模 [2] 和阻抗测量 [4] 提出了挑战。为了解决这个问题，几何声学方法和基于波的方法都广泛假设了简化的局部反应（LR）近似[2,5-11]，它断言边界表面上某个点的响应仅取决于入射到该特定位置的声压，而与周围的声场无关[12]。

为了阐明LR假设的适用范围，人们进行了大量的数值和实验研究。通过研究不同复杂程度的多孔材料内部的波传播模型，发现随着流阻率的减小和材料厚度的增加，LR近似会导致预测的随机入射吸收系数出现更明显的偏差[13]。后来基于分析模型和数值实验的研究[14,15]进一步支持了这些发现。使用频域有限元方法[16-18]对扩散声场进行的室内声学模拟表明，整体平均表面阻抗产生的结果比法向入射阻抗的结果更接近参考值。对于一般的非漫射声场，应用 LR 和 ER 模型揭示了混响衰减曲线和声压级分布方面的差异，正如 Yasuda 等人的分析。 [19]采用多域边界元法。使用组合光束追踪和传输矩阵模型的数值研究表明，LR 模型对于由多层多孔材料组成的表面非常不准确 [20,21]。此外，众所周知，当声波撞击接近掠入射的边界时（通常发生在吊顶周围），ER 和 LR 模型的表现有很大不同 [22]。因此，完成 ER 多孔材料的精确建模以进行室内声学分析非常重要。

为此，一个简单的解决方案是模拟材料内部各个方向的波传播。 Biot 多孔弹性模型[23,24]提供了对空气声波和框架相关弹性波的系统和全面的描述。然而，所有波形的显式计算都是计算密集型的[25,26]。在长波长条件下，即波长明显大于典型孔隙的微观尺寸，可以将具有刚性或软性框架的多孔材料视为等效流体，其特征在于宏观尺度上的有效密度和体积模量[1]，这显着降低了计算量。因此，我们的兴趣仅限于可以通过等效流体模型（EFM）描述的多孔材料。在文献中，EFM，无论是经验的还是现象学的，大多被表述为无理传递函数，例如 Miki 模型 [27] 和 Johnson-Champoux-Allard-Lafarge (JCAL) 模型 [28,29]。为了允许 EFM 的时域分析公式，通常应用临时假设。然而，一些公式[30,31]在有限的频率范围内有效，而其他表示[32,33]涉及带有分数项的卷积核，这在数值离散化和解历史存储方面提出了挑战。

将傅里叶逆变换直接应用于无理性质的 EFM 会产生时域中的分数阶微分算子。为了避免这种情况，现有的多孔介质中时域波传播的数值处理大多以多极有理函数的形式近似频域中EFM的频率相关属性，在信号处理中也称为IIR滤波器。赵等人。 [34]将Z变换应用于IIR滤波器和频域波动方程以避免时域中的卷积积分。另一种方法是通过辅助微分方程 (ADE) 方法对所得时间卷积积分进行数值离散。它将卷积积分在时间上微分，并将其转换为一组附加的辅助变量或记忆变量的一阶常微分方程 (ODE)，可以使用高阶时间积分方案来求解。 Dragna等人将ADE方法应用于Wilson松弛EFM模型[30]来模拟室外声音传播。 [35]。最近，Moufid 等人提出了基于多极的 EFM 时域公式。 [36]用于刚性多孔介质内的波传播。其中，彻底的能量和稳定性分析表明，拟合极点的正性是稳定解的必要条件。最近与使用 ADE 方法对多孔材料进行扩展反应建模相关的工作包括 Refs。 [37-39]。吉田等人。 [38]提出了标量波动方程的隐式时域有限元公式，其中辅助变量以与主要声学变量相同的方式离散化。在[37]中，Pind 等人。将多极近似应用于室内声学的 Miki 模型，其中控制方程通过使用中心通量的不连续 Galerkin 方法进行空间离散。这项工作的一个问题是辅助方程包含主要声学变量的空间导数，因此计算成本随着辅助变量数量的增加而增加。阿洛马尔等人。 [39]通过使用部分分数分解来规避这个问题，类似于其他工作[36,40]，并使用有限差分方案模拟具有扩展反应衬里的流道中的声传播。

在现代建筑设计中，出于卫生、耐用和保护的目的，通常将由纤维或穿孔板制成的附加覆盖物附着在多孔吸声器上，同时，在拓宽的吸收峰值频率范围方面观察到吸声性能的改善[41-43]。典型的例子可以在隔音窗帘和吊顶中找到。广泛的研究工作致力于这些被空气包围的覆盖材料的吸声和传播的数值模拟，例如渗透性薄膜[44-46]和微孔板[46-48]，据作者所知，适用于带有覆盖材料的ER多孔吸声器的数值方案相对较少。由于覆盖物的厚度相对较小，因此这种薄覆盖布可以通过其传输阻抗在声学上表示为压力跃变不连续性，而不是建模为另一个声学域。本工作采用了这种方法，该方法已成功应用于消声器[49]和流道[39]的数值建模。作为 LR 表面，通过利用其传输阻抗增强多孔吸收体的表面阻抗，覆盖物的影响可以轻松地集成到阻抗边界公式中。相比之下，ER 边界公式需要不同传播介质之间适当的界面耦合条件，这在参考文献中针对孔隙弹性材料的时谐分析进行了广泛讨论。 [50,51]。然而，涉及多孔材料和薄覆盖层的界面耦合的时域 ER 边界公式尚未开发出来。

$$
\begin{array}{r} \frac {\partial \mathbf {v}}{\partial t} + \frac {1}{\rho_ {a}} \nabla p = \mathbf {0}, \\ \frac {\partial p}{\partial t} + \rho_ {a} c _ {a} ^ {2} \nabla \cdot \mathbf {v} = 0, \end{array}\tag{1}
$$

(2) 

$$
\begin{array}{r} \mathrm{i} \omega \rho_ {\mathrm{ef}} (\omega) \hat {\mathbf {v}} + \nabla \hat {p} = \mathbf {0}, \\ \mathrm{i} \omega \mathcal {C} _ {\mathrm{ef}} (\omega) \hat {p} + \nabla \cdot \hat {\mathbf {v}} = 0, \end{array}\tag{3a}
$$

(3b) 

$$
\begin{array}{r} \rho_ {\mathrm{ef}} (\omega) \approx \rho_ {m} + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} \frac {B _ {\rho k}}{\zeta_ {\rho k} + \mathrm{i} \omega}, \\ \mathcal {C} _ {\mathrm{ef}} (\omega) \approx \mathcal {C} _ {m} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} \frac {B _ {C k}}{\zeta_ {C k} + \mathrm{i} \omega}, \end{array}\tag{4a}
$$

(4b) 

$$
\mathrm{i} \omega \rho_ {m} \hat {\mathbf {v}} + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} \big (B _ {\rho k} - \frac {B _ {\rho k} \zeta_ {\rho k}}{\zeta_ {\rho k} + \mathrm{i} \omega} \big) \hat {\mathbf {v}} + \nabla \hat {p} = \mathbf {0},\tag{5a}
$$

$$
\mathrm{i} \omega \mathcal {C} _ {m} \hat {p} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} \big (B _ {C k} - \frac {B _ {C k} \zeta_ {C k}}{\zeta_ {C k} + \mathrm{i} \omega} \big) \hat {p} + \nabla \cdot \hat {\mathbf {v}} = 0.\tag{5b}
$$

$$
\begin{array}{r} \rho_ {m} \frac {\partial \mathbf {v}}{\partial t} + \nabla p + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \mathbf {v} - \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \boldsymbol {\phi} _ {\rho k} = \mathbf {0}, \\ \frac {1}{\rho_ {m} c _ {m} ^ {2}} \frac {\partial p}{\partial t} + \nabla \cdot \mathbf {v} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} p - \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \zeta_ {C k} \boldsymbol {\phi} _ {C k} = 0, \end{array}\tag{6a}
$$

(6b) 

$$
\phi_ {\rho k} (\mathbf {x}, t) = \int_ {0} ^ {t} \mathbf {v} (\mathbf {x}, \tau) \mathrm{e} ^ {- \zeta_ {\rho k} (t - \tau)} \mathrm{d} \tau .
$$

$$
\begin{array}{r l} & {\frac {\partial \pmb {\phi} _ {\rho k}}{\partial t} + \zeta_ {\rho k} \pmb {\phi} _ {\rho k} = \mathbf {v}, \quad \forall k \in [ 1, \mathcal {N} _ {\rho} ],} \\ & {\frac {\partial \phi_ {C k}}{\partial t} + \zeta_ {C k} \phi_ {C k} = p, \quad \forall k \in [ 1, \mathcal {N} _ {C} ],} \end{array}\tag{7a}
$$

(7b) 

$$
\begin{array}{r} \mathbf {v} _ {a} \cdot \mathbf {n} _ {a} = - \mathbf {v} _ {m} \cdot \mathbf {n} _ {m}, \\ p _ {a} - p _ {m} = Z _ {t} \mathbf {v} _ {a} \cdot \mathbf {n} _ {a}, \end{array}\tag{8a}
$$

(8b) 

$$
\frac {\partial \mathbf {q}}{\partial t} + \mathbf {A} _ {x} \frac {\partial \mathbf {q}}{\partial x} + \mathbf {A} _ {y} \frac {\partial \mathbf {q}}{\partial y} + \mathbf {A} _ {z} \frac {\partial \mathbf {q}}{\partial z} + \mathbf {D q} = \mathbf {g},\tag{9}
$$

$$
\mathbf {A} _ {j} = \left[ \begin{array}{c c c c} 0 & 0 & 0 & \frac {\delta_ {x j}}{\rho} \\ 0 & 0 & 0 & \frac {\delta_ {y j}}{\rho} \\ 0 & 0 & 0 & \frac {\delta_ {z j}}{\rho} \\ \rho c ^ {2} \delta_ {x j} & \rho c ^ {2} \delta_ {y j} & \rho c ^ {2} \delta_ {z j} & 0 \end{array} \right],
$$

(10) 

$$
\mathbf {D} = \left[ \begin{array}{c c c c} \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 & 0 & 0 \\ 0 & \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 & 0 \\ 0 & 0 & \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 \\ 0 & 0 & 0 & \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \end{array} \right], \quad \mathbf {g} = \left[ \begin{array}{c} \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {x} \\ \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {y} \\ \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {z} \\ \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \zeta_ {C k} \phi_ {C k} \end{array} \right].\tag{11}
$$

$$
\mathbf {q} ^ {e} (\mathbf {x}, t) \approx \mathbf {q} _ {h} ^ {e} (\mathbf {x}, t) = \sum_ {i = 1} ^ {N _ {p}} \mathbf {q} _ {h} ^ {e} (\mathbf {x} _ {i} ^ {e}, t) l _ {i} ^ {e} (\mathbf {x}),\tag{12}
$$

$$
\int_ {\Omega^ {e}} l _ {i} ^ {e} \left(\frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial t} + \mathbf {A} _ {x} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial x} + \mathbf {A} _ {y} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial y} + \mathbf {A} _ {z} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial z} + \mathbf {D} \mathbf {q} _ {h} ^ {e} - \mathbf {g}\right) d \mathbf {x} = \oint_ {\partial \Omega^ {e}} l _ {i} ^ {e} \left(\mathbf {A} _ {n} ^ {e} \mathbf {q} _ {h} ^ {e} - \mathbf {F} ^ {e} (\mathbf {q} _ {h} ^ {e}, \mathbf {q} _ {h} ^ {e +})\right) d \mathbf {x},\tag{13}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/29d3e3a6522a5bbb83c8671c96539b0694d8979ede2eece59f643f95b5883fec.jpg)




$$
\mathbf {q} _ {0} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {q} ^ {-} & \text {if} \quad \mathbf {n} \cdot (\mathbf {x} - \mathbf {x} _ {0}) <   0 \\ \mathbf {q} ^ {+} & \text {if} \quad \mathbf {n} \cdot (\mathbf {x} - \mathbf {x} _ {0}) > 0 \end{array} \right.
$$

$$
\mathbf {A} _ {n} := \mathbf {A} _ {x} n _ {x} + \mathbf {A} _ {y} n _ {y} + \mathbf {A} _ {z} n _ {z} = \mathbf {R} \boldsymbol {\Lambda} \mathbf {R} ^ {- 1},\tag{14}
$$

$$
\mathbf {R} = \frac {1}{2} \left[ \begin{array}{c c c c} - n _ {x} & - 2 n _ {y} & - 2 n _ {z} & n _ {x} \\ - n _ {y} & 2 n _ {x} & 0 & n _ {y} \\ - n _ {z} & 0 & 2 n _ {x} & n _ {z} \\ \rho c & 0 & 0 & \rho c \end{array} \right],\tag{15}
$$

$$
- \lambda_ {i} (\mathbf {q} ^ {\text { neg }} - \mathbf {q} ^ {\text { pos }}) + \mathbf {A} _ {n} (\mathbf {q} ^ {\text { neg }} - \mathbf {q} ^ {\text { pos }}) = \mathbf {0}\tag{16}
$$

$$
\begin{array}{c} c ^ {-} (\mathbf {q} ^ {-} - \mathbf {q} ^ {a}) + \mathbf {A} _ {n} ^ {-} (\mathbf {q} ^ {-} - \mathbf {q} ^ {a}) = \mathbf {0}, \\ \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} - \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {0}, \\ - c ^ {+} (\mathbf {q} ^ {b} - \mathbf {q} ^ {+}) + \mathbf {A} _ {n} ^ {+} (\mathbf {q} ^ {b} - \mathbf {q} ^ {+}) = \mathbf {0}. \end{array}\tag{17a}
$$

(17b) 

(17c) 

$$
\begin{array}{r} \mathbf {q} ^ {-} - \mathbf {q} ^ {a} = \alpha_ {1} \mathbf {r} _ {1} ^ {-}, \\ \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} - \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {0}, \\ \mathbf {q} ^ {b} - \mathbf {q} ^ {+} = \alpha_ {4} \mathbf {r} _ {4} ^ {+}, \end{array}\tag{18a}
$$

(18b) 

(18c) 

$$
\left[ \begin{array}{c} \alpha_ {1} \\ \alpha_ {4} \end{array} \right] = \frac {2}{Z ^ {-} + Z _ {t} + Z ^ {+}} \left[ \begin{array}{c} p ^ {-} - p ^ {+} - Z ^ {+} v _ {n} ^ {+} - (Z _ {t} + Z ^ {+}) v _ {n} ^ {-} \\ p ^ {-} - p ^ {+} + Z ^ {-} v _ {n} ^ {-} + (Z _ {t} + Z ^ {-}) v _ {n} ^ {+} \end{array} \right],\tag{19}
$$

$$
\mathbf {F} = \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} = \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {-} + c ^ {-} \alpha_ {1} \mathbf {r} _ {1} ^ {-},\tag{20}
$$

$$
\mathbf {F} = \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {+} + c ^ {+} \alpha_ {4} \mathbf {r} _ {4} ^ {+}.\tag{21}
$$

$$
\mathbf {F} ^ {a} = \mathbf {R} _ {a} \boldsymbol {\Lambda} _ {a} \left[ \begin{array}{c} \mathcal {R} _ {a m} \varpi_ {a} ^ {o} + \mathcal {T} _ {m a} \varpi_ {m} ^ {o} \\ 0 \\ 0 \\ \varpi_ {a} ^ {o} \end{array} \right],\tag{22}
$$

$$
\begin{array}{r} \varpi_ {a} ^ {o} = \frac {p _ {a}}{Z _ {a}} + \mathbf {v} _ {a} \cdot \mathbf {n} _ {a}, \\ \varpi_ {m} ^ {o} = \frac {p _ {m}}{Z _ {m}} + \mathbf {v} _ {m} \cdot \mathbf {n} _ {m}, \end{array}\tag{23a}
$$

(23b) 

$$
\mathcal {R} _ {a m} = \frac {Z _ {t} + Z _ {m} - Z _ {a}}{Z _ {t} + Z _ {m} + Z _ {a}}, \quad \mathcal {T} _ {m a} = \frac {2 Z _ {m}}{Z _ {t} + Z _ {m} + Z _ {a}},\tag{24}
$$

$$
\mathbf {F} ^ {m} = \mathbf {R} _ {m} \boldsymbol {\Lambda} _ {m} \left[ \begin{array}{c} \mathcal {R} _ {m a} \varpi_ {m} ^ {o} + \mathcal {T} _ {a m} \varpi_ {a} ^ {o} \\ 0 \\ 0 \\ \varpi_ {m} ^ {o} \end{array} \right],\tag{25}
$$

$$
\mathcal {R} _ {m a} = \frac {Z _ {t} + Z _ {a} - Z _ {m}}{Z _ {t} + Z _ {m} + Z _ {a}}, \quad \mathcal {T} _ {a m} = \frac {2 Z _ {a}}{Z _ {t} + Z _ {m} + Z _ {a}}.\tag{26}
$$

$$
\mathbf {F} ^ {e} (\mathbf {q} _ {h} ^ {e}, \mathbf {q} _ {h} ^ {e +}) = \mathbf {R} _ {a} \boldsymbol {\Lambda} _ {a} \left[ \begin{array}{c} \frac {p _ {h} ^ {e +}}{Z _ {a}} - \mathbf {v} _ {h} ^ {e +} \cdot \mathbf {n} _ {e} \\ 0 \\ 0 \\ \frac {p _ {h} ^ {e}}{Z _ {a}} + \mathbf {v} _ {h} ^ {e} \cdot \mathbf {n} _ {e} \end{array} \right].\tag{27}
$$

$$
Z _ {t} (\omega) = \left(\frac {1}{r _ {f}} + \frac {1}{\mathrm{i} \omega m}\right) ^ {- 1}.\tag{28}
$$

$$
\mathcal {R} _ {a m} = 1 - Z _ {a} B _ {t} + \frac {1}{\mathrm{i} \omega + \zeta_ {t}} (Z _ {a} B _ {t} \zeta_ {t} - \frac {Z _ {a} r _ {f}}{m})\tag{29a}
$$

$$
\mathcal {T} _ {a m} = Z _ {a} B _ {t} - \frac {1}{\mathrm{i} \omega + \zeta_ {t}} (Z _ {a} B _ {t} \zeta_ {t} - \frac {Z _ {a} r _ {f}}{m})\tag{29b}
$$

$$
\mathcal {R} _ {m a} = 1 - Z _ {m} B _ {t} + \frac {1}{\mathrm{i} \omega + \zeta_ {t}} (Z _ {m} B _ {t} \zeta_ {t} - \frac {Z _ {m} r _ {f}}{m})\tag{29c}
$$

$$
\mathcal {T} _ {m a} = Z _ {m} B _ {t} - \frac {1}{\mathrm{i} \omega + \zeta_ {t}} (Z _ {m} B _ {t} \zeta_ {t} - \frac {Z _ {m} r _ {f}}{m}),\tag{29d}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {v} _ {x h} ^ {e}}{\partial t} + \frac {1}{\rho_ {a}} \mathbf {S} _ {x} ^ {e} \mathbf {p} _ {h} ^ {e} = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \left(\frac {n _ {x} ^ {e}}{\rho_ {a}} \mathbf {p} _ {h} ^ {e} - \hat {\mathbf {F}} _ {v _ {x}} ^ {e r}\right),\tag{30a}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {v} _ {y h} ^ {e}}{\partial t} + \frac {1}{\rho_ {a}} \mathbf {S} _ {y} ^ {e} \mathbf {p} _ {h} ^ {e} = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \bigg (\frac {n _ {y} ^ {e}}{\rho_ {a}} \mathbf {p} _ {h} ^ {e} - \hat {\mathbf {F}} _ {v _ {y}} ^ {e r} \bigg),\tag{30b}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {v} _ {z h} ^ {e}}{\partial t} + \frac {1}{\rho_ {a}} \mathbf {S} _ {z} ^ {e} \mathbf {p} _ {h} ^ {e} = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \left(\frac {n _ {z} ^ {e}}{\rho_ {a}} \mathbf {p} _ {h} ^ {e} - \hat {\mathbf {F}} _ {v _ {z}} ^ {e r}\right),\tag{30c}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {p} _ {h} ^ {e}}{\partial t} + \rho_ {a} c _ {a} ^ {2} \bigg (\mathbf {S} _ {x} ^ {e} \mathbf {v} _ {x h} ^ {e} + \mathbf {S} _ {y} ^ {e} \mathbf {v} _ {y h} ^ {e} + \mathbf {S} _ {z} ^ {e} \mathbf {v} _ {z h} ^ {e} \bigg) = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \bigg (\rho_ {a} c _ {a} ^ {2} \big (\mathbf {v} _ {x h} ^ {e} n _ {x} ^ {e} + \mathbf {v} _ {y h} ^ {e} n _ {y} ^ {e} + \mathbf {v} _ {z h} ^ {e} n _ {z} ^ {e} \big) - \hat {\mathbf {F}} _ {p} ^ {e r} \bigg),\tag{30d}
$$

$$
\mathbf {M} _ {m n} ^ {e} = \int_ {\Omega^ {e}} l _ {m} ^ {e} (\mathbf {x}) l _ {n} ^ {e} (\mathbf {x}) \mathrm{d} \mathbf {x} \quad \in \mathbb {R} ^ {N _ {p} \times N _ {p}},
$$

(31a) 

$$
(\mathbf {S} _ {j} ^ {e}) _ {m n} = \int_ {\Omega^ {e}} l _ {m} ^ {e} (\mathbf {x}) \frac {\partial l _ {n} ^ {e} (\mathbf {x})}{\partial x _ {j}} \mathrm{d} \mathbf {x} \quad \in \mathbb {R} ^ {N _ {p} \times N _ {p}},\tag{31b}
$$

$$
\mathbf {M} _ {m n} ^ {e r} = \int_ {\partial \Omega^ {e r}} l _ {m} ^ {e r} (\mathbf {x}) l _ {n} ^ {e r} (\mathbf {x}) \mathrm{d} \mathbf {x} \quad \in \mathbb {R} ^ {N _ {p} \times N _ {f p}},\tag{31c}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {v} _ {x h} ^ {e}}{\partial t} + \frac {1}{\rho_ {m}} \mathbf {S} _ {x} ^ {e} \mathbf {p} _ {h} ^ {e} = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \left(\frac {n _ {x} ^ {e}}{\rho_ {m}} \mathbf {p} _ {h} ^ {e} - \hat {\mathbf {F}} _ {v _ {x}} ^ {e r}\right) - \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \mathbf {M} ^ {e} \left(\mathbf {v} _ {x h} ^ {e} - \zeta_ {\rho k} \boldsymbol {\phi} _ {\rho k} ^ {x e}\right),\tag{32a}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {v} _ {y h} ^ {e}}{\partial t} + \frac {1}{\rho_ {m}} \mathbf {S} _ {y} ^ {e} \mathbf {p} _ {h} ^ {e} = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \left(\frac {n _ {y} ^ {e}}{\rho_ {m}} \mathbf {p} _ {h} ^ {e} - \hat {\mathbf {F}} _ {v _ {y}} ^ {e r}\right) - \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \mathbf {M} ^ {e} \left(\mathbf {v} _ {y h} ^ {e} - \zeta_ {\rho k} \boldsymbol {\phi} _ {\rho k} ^ {y e}\right),\tag{32b}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {v} _ {z h} ^ {e}}{\partial t} + \frac {1}{\rho_ {m}} \mathbf {S} _ {z} ^ {e} \mathbf {p} _ {h} ^ {e} = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \left(\frac {n _ {z} ^ {e}}{\rho_ {m}} \mathbf {p} _ {h} ^ {e} - \hat {\mathbf {F}} _ {v _ {z}} ^ {e r}\right) - \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \mathbf {M} ^ {e} \left(\mathbf {v} _ {z h} ^ {e} - \zeta_ {\rho k} \boldsymbol {\phi} _ {\rho k} ^ {z e}\right),\tag{32c}
$$

$$
\mathbf {M} ^ {e} \frac {\partial \mathbf {p} _ {h} ^ {e}}{\partial t} + \rho_ {m} c _ {m} ^ {2} \left(\mathbf {S} _ {x} ^ {e} \mathbf {v} _ {x h} ^ {e} + \mathbf {S} _ {y} ^ {e} \mathbf {v} _ {y h} ^ {e} + \mathbf {S} _ {z} ^ {e} \mathbf {v} _ {z h} ^ {e}\right) = \sum_ {r = 1} ^ {f} \mathbf {M} ^ {e r} \left(\rho_ {m} c _ {m} ^ {2} \left(\mathbf {v} _ {x h} ^ {e} n _ {x} ^ {e} + \mathbf {v} _ {y h} ^ {e} n _ {y} ^ {e} + \mathbf {v} _ {z h} ^ {e} n _ {z} ^ {e}\right) - \hat {\mathbf {F}} _ {p} ^ {e r}\right)
$$

$$
- \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \mathbf {M} ^ {e} \bigg (\mathbf {p} _ {h} ^ {e} - \zeta_ {C k} \boldsymbol {\phi} _ {C k} ^ {e} \bigg),\tag{32d}
$$

$$
\frac {\partial \tilde {\mathbf {q}} _ {h}}{\partial t} = \mathcal {L} \big (\tilde {\mathbf {q}} _ {h} (t), t \big),\tag{33}
$$

$$
\tilde {\mathbf {q}} _ {h} (t + \Delta t) = \tilde {\mathbf {q}} _ {h} (t) + \sum_ {i = 1} ^ {N _ {t}} \frac {\Delta t ^ {i}}{i !} \mathcal {L} ^ {i} \tilde {\mathbf {q}} _ {h} (t),\tag{34}
$$

$$
\Delta t = C _ {C F L} \Delta x _ {l} \frac {1}{c} \frac {1}{(2 N + 1)},\tag{35}
$$

$$
\begin{array}{r l} & {\rho_ {\mathrm{ef}} (\omega) = \frac {\rho_ {a} \alpha_ {\infty}}{\varphi} \Big [ 1 + \frac {\sigma \varphi}{\mathrm{i} \omega \alpha_ {\infty} \rho_ {a}} \big (1 + \frac {4 \mathrm{i} \alpha_ {\infty} ^ {2} \eta \rho_ {a}}{\sigma^ {2} \Lambda^ {2} \varphi^ {2}} \big) ^ {1 / 2} \Big ],} \\ & {\mathcal {C} _ {\mathrm{ef}} (\omega) = \frac {\varphi}{\rho_ {a} c _ {a} ^ {2}} \Bigg (\gamma - \frac {\gamma - 1}{\big [ 1 + \frac {\varphi \eta}{\mathrm{i} \omega k _ {0} ^ {\prime} \rho_ {a} P _ {r}} (1 + \frac {4 \mathrm{i} \omega k _ {0} ^ {\prime 2} \rho_ {a} P _ {r}}{\eta \Lambda^ {\prime 2} \varphi^ {2}}) ^ {1 / 2} \big ]} \Bigg).} \end{array}\tag{36a}
$$

(36b) 



![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/77b6caf1c45e9c315e7636c70a254a6b5c3d64f6b4938ef31550a568b099214f.jpg)



$$
\begin{array}{r l} & p (x, t = 0) = \big (1 - \big [ \frac {x - x _ {s}}{B} \big ] ^ {2} \big) \mathrm{e} ^ {- (\frac {x - x _ {s}}{\sqrt {2} B}) ^ {2}}, \\ & u (x, t = 0) = \frac {1}{\rho_ {a} c _ {a}} \big (1 - \big [ \frac {x - x _ {s}}{B} \big ] ^ {2} \big) \mathrm{e} ^ {- (\frac {x - x _ {s}}{\sqrt {2} B}) ^ {2}}, \end{array}\tag{37a}
$$

(37b) 

$$
\epsilon_ {\mathrm{num}} (\bar {t}) = \frac {\| p _ {\mathrm{ana*}} (\bar {t}) - p _ {\mathrm{num}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana*}} (\bar {t}) \| _ {L ^ {2}}}\tag{38}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/efe34afcf5532b448fc874208cafc29467f9a5c54bfcc554652d548cd6542686.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/a4a5f9568f306a880462f2d6b609d1d7673934192eac686a2d0682308a7bd1e7.jpg)



(b)




$$
\epsilon_ {\mathrm{model}} (\bar {t}) = \frac {\| p _ {\mathrm{ana*}} (\bar {t}) - p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}},\tag{39}
$$

$$
\epsilon_ {\mathrm{tot}} (\bar {t}) = \frac {\| p _ {\mathrm{ana}} (\bar {t}) - p _ {\mathrm{num}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}}\tag{40}
$$

$$
\epsilon_ {\mathrm{num}} ^ {l} (\bar {t}) = \frac {\left(\int_ {0} ^ {\bar {t}} | \lceil p \rceil_ {\mathrm{num}} - \lceil p \rceil_ {\mathrm{ana*}} | ^ {2} \mathrm{d} t\right) ^ {1 / 2}}{\left(\int_ {0} ^ {\bar {t}} \lceil p \rceil_ {\mathrm{ana*}} ^ {2} \mathrm{d} t\right) ^ {1 / 2}},\tag{41}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/2aac02e1de25a618b97144787e390ead1e0c5424968ad2b9553244a90bf4af1c.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/dd042b9477c9bbefdbeb06c111e65102809a41abf685c076dfc1b4fea2f13d32.jpg)



(b)




![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/5a902df616ae1c1e7d0a57acc262ecfeb134ca8667b275a4469f18820b7b78bf.jpg)



$$
p (\mathbf {x}, t = 0) = \mathrm{e} ^ {\frac {- \ln 2}{b ^ {2}} (\mathbf {x} - \mathbf {x} _ {s}) ^ {2}},\tag{42a}
$$

$$
\mathbf {v} (\mathbf {x}, t = 0) = \mathbf {0},\tag{42b}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/2923e223c00c7d75c7238a955e8719375cbce39086f387ecef6d7c51d2009a8f.jpg)




![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/28ba7192d4c72db2cc80e564a357f081b8a77453b2e116fe7cbb5fb6c4e7b234.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/5d6dd3e1e6d727ae0207482e788b0ae07bf0a37e15d106761dac1e07b8eb69de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0d0dddcf450d1e6fc02f434169477afcde0eede08e85597df95dbbcf5f46071c.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/be5a701817745e9df320538fb93bd21f3329d55da075c7cdf27ddbdf3412de75.jpg)



(b)



图 7. 三聚氰胺泡沫上方的模拟和分析压力场：(a) $\mathbf { x } _ { r 1 }, \theta ^ { \circ } = 0 ^ { \circ }$，(b) $\mathbf { x } _ { r 2 }, \theta ^ { \circ } = 8 0 ^ { \circ }$

半带宽值为 $b = 0 . 1 \textrm { m }$，表示源频谱高达 2 kHz。整个计算域的维度为 $[ - 3, 4 ] \times [ - 3, 4 ] \times [ - d, 3 . 5 ]$ m，其中 ??是多孔材料的厚度。使用网格划分软件 <sup>GMSH</sup> [77] 为空气和多孔材料生成符合四面体网格的非结构化几何形状。这里，网格尺寸受到多孔材料厚度的限制，该厚度等于四面体单元一侧的长度。为了获得足够的空间分辨率，使用七阶多项式基函数 $( N = 7 )$ 进行空间离散化，从而在 2 kHz 下每个波长产生大约 9 个体积平均自由度。时间顺序设置为 $N _ { t } = 5$ 以在模拟精度和效率之间取得良好的平衡。内切球的最小半径 min $( r _ { i n } )$ 用作单元尺寸 $\varDelta x _ { l }$ 的度量。在进行的数值试验中，单元尺寸是时间步长的主要限制因素。硬壁边界条件施加于整个计算域的外部边界，并且在来自外部边界的寄生反射到达之前接收器位置记录的压力信号被切断。 

图 7 显示了模拟和分析压力谱 $\hat { p }$ 在振幅方面的比较 | ̂??|对于具有两种不同厚度的三聚氰胺泡沫的情况，相$\vartheta ( \hat { p } )$。解析解是通过使用 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } }$ 的精确值作为输入来计算的，而数值解是基于具有 5 个实极点的 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } }$ 的近似值。使用初始高斯脉冲的解析自由场时间解来标准化其非平坦源功率谱。模拟结果的大小已标准化，使得自由场解的形式为 $\mathrm { e } ^ { - \mathrm { i } k r } / ( 4 \pi r )$ 。同样，图8显示了当多孔吸收体由表1所示的玻璃棉表示时的比较结果。从这些结果可以看出，模拟解和参考解之间达到了很好的一致性，证明了所提出的边界方案在3D空间中的适用性以及在宽频率范围内的高精度。

$$
\epsilon_ {\mathrm{amp}} (f) = 2 0 \log_ {1 0} \left| \frac {\hat {p} _ {\mathrm{num}} (f)}{\hat {p} _ {\mathrm{ana}} (f)} \right|,\tag{43a}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/f119484f68d86af3d00fd29c135eb83ce81257ac3c7ebdd92f1f865920259200.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/69e88d687eee10175ce7416ad49e56ee9980d9239eceec444d4df515f1f783d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0680e79aba9d688dca2ae324c8dc6f010da8a23ae4d942d6dc8a59c9c660915c.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/bb626ceace6fe15218e1d69747cbf0f3395cb08b748c23a32c87735acf0096b1.jpg)



(b)




![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/d2e9021351f062a4bfa038b9265e0359ae21ceaed40603481deacb29fb424f79.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/66ef5c2060a1ef0236d3d86b8632acb2cb9d6b98b9a6226b6b862ec7164b08d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0bf916740db2f59de5341a1620dbae12f919fe0d4ba019d3b98a04e0e242ee6d.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/fd94b83c6ce20012c6178cea7b6ecca1f5da4a54f48bc714ad195f6680eec654.jpg)



(b)




$$
\epsilon_ {\vartheta} (f) = \frac {1}{\pi} \big (\vartheta \big (\hat {p} _ {\mathrm{num}} (f) \big) - \vartheta \big (\hat {p} _ {\mathrm{ana}} (f) \big) \big) \times 100 \%.\tag{43b}
$$

图 9 显示耗散误差 ??和相位误差？？两种多孔吸收体的模拟结果，其具有明显不同的 $\epsilon _ { a m p }$ $\epsilon _ { \vartheta }$ 不同的流阻值 $\sigma .$ 。网格和多项式阶数的组合导致 2 kHz 下每个波长 9 个自由度的空间分辨率。正如预期的那样，当空间分辨率足够时，数值误差保持在可接受的水平。值得注意的是，低频范围（500 Hz 以下）的误差大于中频范围（500 – 1500 Hz）。这是由于记录的压力信号过早被切断，导致反射声场的低频功率损失。由于斜入射情况的第二个接收器 $\mathbf { x } _ { r 2 }$ 比第一个接收器更靠近外部硬壁边界，因此过早切割发生得更快。因此，图9(b)中的误差比图9(a)中的误差大。 

## 5. 结论

在这项工作中，为了时域室内声学模拟的目的，开发了一个用于模拟多孔材料（包括薄覆盖物）的一般扩展反应边界的数值框架。等效流体模型用于描述任意多孔材料内的声波传播，其有效密度和压缩率可以使用多极有理函数很好地近似。通过应用ADE方法计算卷积积分，所得到的多孔材料时域控制方程可以写成统一的双曲线形式，就像无损空气的线性声学方程一样。基于解决潜在的黎曼问题，开发了一致的迎风数值通量公式，确保传播介质（包括空气、覆盖材料和多孔吸收体）之间适当的物理耦合。柔软渗透膜模型用于表征覆盖材料的声学特性。为了解决由于时间步长受限而导致的电位低效问题，采用了局部时间步长方案 

一维数值测试验证了所提出公式的收敛性，其中获得了 DG 方法的最佳收敛速率 $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$。同时，证明了接口耦合不会产生额外的误差。三维测试进一步验证了所提出的方法在多维情况下表示实际扩展反应阻抗边界的能力。幅度和相位信息均被准确捕获。



















































































































































































































































































































