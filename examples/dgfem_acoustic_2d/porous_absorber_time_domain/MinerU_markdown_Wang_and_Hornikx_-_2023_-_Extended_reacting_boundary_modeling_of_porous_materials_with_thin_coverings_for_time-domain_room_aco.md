# 具有薄覆盖物的多孔材料的扩展反应边界建模，用于时域室内声学模拟

# Extended reacting boundary modeling of porous materials with thin coverings for time-domain room acoustic simulations

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/61067af0e10238ed13ffe85864db96029c4beddd791f2d1d0ccc21153ff064a2.jpg)

王惠清<sup>*</sup>，Maarten Hornikx

Huiqing Wang <sup>∗</sup>, Maarten Hornikx

建筑声学，建筑环境系，埃因霍温科技大学，P.O. Box 513, 5600 MB 埃因霍温, 荷兰

Building Acoustics, Department of the Built Environment, Eindhoven University of Technology, P.O. Box 513, 5600 MB Eindhoven, The Netherlands

文章信息

ARTICLE INFO

关键词： 时域室内声学建模 多孔层的扩展反应 薄覆盖材料 高阶精度 精确黎曼求解器

Keywords: Time-domain room acoustic modeling Extended reaction of porous layers Thin covering materials High-order accuracy Exact Riemann solver

## 摘要

## A B S T R A C T

声学边界条件建模对房间声学模拟的准确性有重大影响，在室内建筑环境的设计阶段以提高声学舒适度方面发挥着重要作用。在这项工作中，提出了一种基于不连续伽辽金（DG）方法的数值框架，用于对薄材料覆盖的多孔吸收体的扩展反应边界进行建模。通过将多孔材料视为子域来应用域分解方法。等效流体模型用于描述多孔材料的声学特性，其有效密度和压缩率作为无理函数由频域中的多极有理函数近似。通过采用辅助微分方程方法计算时间卷积，多孔材料的增广时域控制方程可以用与线性声学方程相同的统一双曲形式表示，这进一步实现了整个域内一致的迎风数值通量公式。通过解决基本的黎曼问题来处理传播介质之间界面上的数值耦合。与用于室内声学扩展反应边界建模的现有 DG 方法相比，导出的迎风数值通量公式不涉及辅助变量的计算。所提出的框架产生了一个适定的线性双曲系统，其边界条件由“均匀 Kreiss 条件”（Kreiss，1970）指导。通过考虑柔软渗透膜模型来说明覆盖材料的声学特性。利用局部时间步进方法来提高计算效率。针对一维解析解进行数值验证，以验证所需的高阶收敛速度。球面波前建模的 3D 案例研究证明了该公式的宽带精度。

Modeling of acoustic boundary conditions has a significant impact on the accuracy of room acoustic simulations, which play an important role in the design phase of indoor built environments in order to improve the acoustical comfort. In this work, a numerical framework based on the discontinuous Galerkin (DG) method is presented for modeling extended reacting boundaries of porous absorbers covered by thin materials. The domain decomposition methodology is applied by treating the porous material as a subdomain. Equivalent fluid models are used to depict the acoustic properties of porous materials, whose effective density and compressibility as irrational functions are approximated by multipole rational functions in the frequency domain. By employing the auxiliary differential equation approach to calculate the time convolution, the augmented time-domain governing equations of porous materials can be expressed in the same unified hyperbolic form as the linear acoustic equations, which further enables a consistent upwind numerical flux formulation throughout the whole domain. The numerical coupling across the interface between propagation media is handled by solving the underlying Riemann problem. Compared to existing approaches with the DG method for extended reacting boundaries modeling for room acoustics, the derived upwind numerical flux formulation does not involve the computation of auxiliary variables. The presented framework yields a well-posed linear hyperbolic system with admissible boundary conditions as guided by the ‘‘uniform Kreiss condition’’ (Kreiss, 1970). Acoustic properties of the covering materials are illustrated by considering a limp permeable membrane model. A local time-stepping approach is utilized to improve computational efficiency. Numerical validations against analytical solutions in 1D are performed to verify the desired high-order convergence rate. A 3D case study on modeling spherical wave fronts demonstrates the broadband accuracy of the formulation.

## 1. 引言

## 1. Introduction

由于多孔材料具有宽带吸声能力和成本效益，因此对于各种声学应用至关重要。例如，它们可以作为边界表面的必要声学处理，以提高建筑环境的声学舒适度。因此，对多孔吸声体附近声波传播的精确建模一直是正在进行的研究的主题[1]。在室内声学建模中，多孔材料的吸声特性通常通过几何声学方法 [2] 的吸收系数或基于波的方法 [3] 的表面阻抗来表征。这些特性从根本上取决于入射波的频率和角度。表现出所谓的扩展反应 (ER) 行为，这对严格的数值建模 [2] 和阻抗测量 [4] 提出了挑战。为了解决这个问题，几何声学方法和基于波的方法都广泛假设了简化的局部反应（LR）近似[2,5-11]，它断言边界表面上某个点的响应仅取决于入射到该特定位置的声压，而与周围的声场无关[12]。

Porous materials are essential for a variety of applications in acoustics due to their broadband sound absorption capabilities and cost-effectiveness. For instance, they serve as necessary acoustic treatments of boundary surfaces in order to improve the acoustic comfort of built environments. Therefore, accurate modeling of sound wave propagation in the immediate vicinity of porous absorbers has been a subject of ongoing research [1]. In room acoustic modeling, acoustic absorption properties of porous materials are typically characterized by absorption coefficients for geometrical acoustic methods [2] or surface impedances for wave-based methods [3]. These properties fundamentally depend on the frequency and the angle of incidence waves. exhibiting the so-called extended reacting (ER) behavior that poses challenges for both rigorous numerical modeling [2] and impedance measurements [4]. To address this issue, a simplifying local reacting (LR) approximation is extensively assumed for both geometrical acoustic methods and wave-based methods [2,5–11], which asserts that the response at a certain point on the boundary surface is dependent only on the sound pressure incident on that specific location regardless of the surrounding acoustic field [12].

为了阐明LR假设的适用范围，人们进行了大量的数值和实验研究。通过研究不同复杂程度的多孔材料内部的波传播模型，发现随着流阻率的减小和材料厚度的增加，LR近似会导致预测的随机入射吸收系数出现更明显的偏差[13]。后来基于分析模型和数值实验的研究[14,15]进一步支持了这些发现。使用频域有限元方法[16-18]对扩散声场进行的室内声学模拟表明，整体平均表面阻抗产生的结果比法向入射阻抗的结果更接近参考值。对于一般的非漫射声场，应用 LR 和 ER 模型揭示了混响衰减曲线和声压级分布方面的差异，正如 Yasuda 等人的分析。 [19]采用多域边界元法。使用组合光束追踪和传输矩阵模型的数值研究表明，LR 模型对于由多层多孔材料组成的表面非常不准确 [20,21]。此外，众所周知，当声波撞击接近掠入射的边界时（通常发生在吊顶周围），ER 和 LR 模型的表现有很大不同 [22]。因此，完成 ER 多孔材料的精确建模以进行室内声学分析非常重要。

To elucidate the applicable range of the LR assumption, numerous numerical and experimental studies have been conducted. By investigating wave propagation models inside porous materials of varying degrees of complexity, it was found that the LR approximation results in more noticeable deviations in terms of predicted random incidence absorption coefficients as the flow resistivity decreases and the thickness of materials increases [13]. Later studies [14,15] based on both analytical models and numerical experiments further support these findings. Room acoustic simulations of a diffuse sound field using the frequency-domain finite element method [16–18] have shown that the ensemble-averaged surface impedance yields results that are comparably closer to reference values than those with the normal incidence impedance. For non-diffuse sound fields in general, applying LR and ER models reveals discrepancies in terms of reverberation decay curves and sound pressure level distributions, as analyzed by Yasuda et al. [19] with the multi-domain boundary element method. Numerical studies using the combined beam-tracing and transfer-matrix model showed that the LR model is highly inaccurate for surfaces consisting of multilayer porous materials [20,21]. Furthermore, it is well understood that ER and LR models behave considerably differently when the sound wave impinges on boundaries close to grazing incidence, as typically occurs around suspended ceilings [22]. Therefore, it is important to accomplish accurate modeling of ER porous materials for room acoustic analysis.

为此，一个简单的解决方案是模拟材料内部各个方向的波传播。 Biot 多孔弹性模型[23,24]提供了对空气声波和框架相关弹性波的系统和全面的描述。然而，所有波形的显式计算都是计算密集型的[25,26]。在长波长条件下，即波长明显大于典型孔隙的微观尺寸，可以将具有刚性或软性框架的多孔材料视为等效流体，其特征在于宏观尺度上的有效密度和体积模量[1]，这显着降低了计算量。因此，我们的兴趣仅限于可以通过等效流体模型（EFM）描述的多孔材料。在文献中，EFM，无论是经验的还是现象学的，大多被表述为无理传递函数，例如 Miki 模型 [27] 和 Johnson-Champoux-Allard-Lafarge (JCAL) 模型 [28,29]。为了允许 EFM 的时域分析公式，通常应用临时假设。然而，一些公式[30,31]在有限的频率范围内有效，而其他表示[32,33]涉及带有分数项的卷积核，这在数值离散化和解历史存储方面提出了挑战。

To that end, a straightforward solution is to simulate wave propagation in all directions inside the material. The Biot poroelastic model [23,24] provides systematic and comprehensive descriptions of both the airborne acoustic wave and the frame-associated elastic waves. However, explicit calculations of all wave forms are computationally intensive [25,26]. Under the long-wavelength condition, i.e., the wavelength is significantly larger than microscopic sizes of typical pores, it is justifiable to view porous material with a rigid or limp frame as an equivalent fluid characterized by its effective density and bulk modulus on a macroscopic scale [1], which reduces the calculation loads remarkably. Therefore, our interest is restricted to porous materials that can be described by the equivalent fluid model (EFM). In the literature, EFMs, either empirical or phenomenological, are mostly formulated as irrationa transfer functions, e.g., the Miki model [27] and the Johnson–Champoux–Allard–Lafarge (JCAL) model [28,29]. To allow analytical time-domain formulations of the EFM, ad hoc assumptions are usually applied. However, some formulations [30,31] are valid to a limited frequency range, and other representations [32,33] involve convolution kernels with fractional terms that pose challenges in terms of numerical discretization and storage of solution history.

将傅里叶逆变换直接应用于无理性质的 EFM 会产生时域中的分数阶微分算子。为了避免这种情况，现有的多孔介质中时域波传播的数值处理大多以多极有理函数的形式近似频域中EFM的频率相关属性，在信号处理中也称为IIR滤波器。赵等人。 [34]将Z变换应用于IIR滤波器和频域波动方程以避免时域中的卷积积分。另一种方法是通过辅助微分方程 (ADE) 方法对所得时间卷积积分进行数值离散。它将卷积积分在时间上微分，并将其转换为一组附加的辅助变量或记忆变量的一阶常微分方程 (ODE)，可以使用高阶时间积分方案来求解。 Dragna等人将ADE方法应用于Wilson松弛EFM模型[30]来模拟室外声音传播。 [35]。最近，Moufid 等人提出了基于多极的 EFM 时域公式。 [36]用于刚性多孔介质内的波传播。其中，彻底的能量和稳定性分析表明，拟合极点的正性是获得稳定解的必要条件。最近与使用 ADE 方法对多孔材料进行扩展反应建模相关的工作包括 Refs。 [37-39]。吉田等人。 [38]提出了标量波动方程的隐式时域有限元公式，其中辅助变量以与主要声学变量相同的方式离散化。在[37]中，Pind 等人。将多极近似应用于室内声学的 Miki 模型，其中控制方程通过使用中心通量的不连续 Galerkin 方法进行空间离散。这项工作的一个问题是辅助方程包含主要声学变量的空间导数，因此计算成本随着辅助变量数量的增加而增加。阿洛马尔等人。 [39]通过使用部分分数分解来规避这个问题，类似于其他工作[36,40]，并使用有限差分方案模拟具有扩展反应衬里的流道中的声传播。

Direct application of the inverse Fourier transform to the EFMs of irrational nature produces fractional differential operators in the time domain. To avoid that, existing numerical treatments of time-domain wave propagation in porous media mostly approximate the frequency-dependent attributes of EFMs in the frequency domain in the form of multipole rational functions, which are also known as IIR filters in signal processing. Zhao et al. [34] applied the Z-transform to the IIR filters and frequency-domain wave equations in order to avoid the convolution integrals in the time domain. Another approach is to numerically discretize the resulting time convolution integrals by the auxiliary differential equation (ADE) method. It differentiates the convolution integrals in time and transforms them into an additional set of first-order ordinary differential equations (ODEs) of auxiliary or memory variables, which can be solved with a high-order time integration scheme. The ADE method was applied to the Wilson’s relaxation EFM model [30] to simulate outdoor sound propagation by Dragna et al. [35]. More recently, a multipole based time-domain formulation of the EFM was presented by Moufid et al. [36] for wave propagation inside rigid porous media. Therein, a thorough energy and stability analysis showed that the positivity of fitting poles are necessary conditions to have stable solutions. More recent works relevant to the extended reaction modeling of porous materials using the ADE method include Refs. [37–39]. Yoshida et al. [38] presented an implicit time-domain finite element formulation for the scalar wave equation, where the auxiliary variables are discretized in the same manner as the primary acoustic variables. In [37], Pind et al. applied the multipole approximation to the Miki model for room acoustics, where the governing equations are spatially discretized by the discontinuous Galerkin method using the central flux. One issue with this work is that the auxiliary equations contain spatial derivatives of the principal acoustic variables, and consequently the computational cost increases with the number of auxiliary variables. Alomar et al. [39] circumvented this issue by using the partial fractional decomposition, similar to other works [36,40], and simulated acoustic propagation in flow ducts with extended-reacting liners using the finite difference scheme.

在现代建筑设计中，出于卫生、耐用和保护的目的，通常将由纤维或穿孔板制成的附加覆盖物附加到多孔吸声器上，同时观察到在拓宽吸收峰值频率范围方面改善了吸声性能[41-43]。典型的例子可以在隔音窗帘和吊顶中找到。广泛的研究工作致力于这些被空气包围的覆盖材料的吸声和传播的数值模拟，例如渗透性薄膜[44-46]和微孔板[46-48]，据作者所知，适用于带有覆盖材料的ER多孔吸声器的数值方案相对较少。由于覆盖物的厚度相对较小，因此这种薄覆盖布可以通过其传输阻抗在声学上表示为压力跳跃不连续性，而不是被建模为另一个声学域。本工作采用了这种方法，该方法已成功应用于消声器[49]和流道[39]的数值建模。作为 LR 表面，通过利用其传递阻抗增强多孔吸收体的表面阻抗，覆盖物的影响可以轻松地集成到阻抗边界公式中。相比之下，ER 边界公式需要不同传播介质之间适当的界面耦合条件，这在参考文献中针对孔隙弹性材料的时谐分析进行了广泛讨论。 [50,51]。然而，涉及多孔材料和薄覆盖层的界面耦合的时域 ER 边界公式尚未开发出来。

In modern architectural designs, additional coverings made from fibers or perforated plates are typically attached to porous absorbers for purposes of hygiene, durability, and protection, Meanwhile, improved sound absorption performance in terms of the broadened absorption peak frequency range are observed [41–43]. Typical examples can be found in acoustic curtains and suspended ceilings. Extensive research efforts have been devoted to numerical modeling of sound absorption and transmission of these covering materials surrounded by air, such as permeable thin membranes [44–46] and microperforated panels [46–48], numerical schemes suitable for ER porous absorbers with covering materials are relatively rare to the authors' knowledge. Since the thickness of the cover is relatively small this thin covering cloth can be acoustically represented by its transfer impedance as a pressure jump discontinuity instead of being modeled as another acoustic domain. This approach, which has been successfully applied in numerical modeling of mufflers [49] and flow ducts [39], is adopted in this work. Being a LR surface, the influence of coverings can be easily integrated into the impedance boundary formulation by augmenting the surface impedance of porous absorbers with its transfer impedance. In contrast, an ER boundary formulation necessitates an appropriate interface coupling condition between different propagation media, which was discussed extensively for the time-harmonic analysis of pore-elastic materials in Refs. [50,51]. However, a time-domain ER boundary formulation for interface coupling involving porous materials and thin coverings has not been developed.

这项工作的主要重点是在时域不连续伽辽金方法的框架内提出一种用于扩展多孔材料反应边界的时域公式。所提出的公式用于室内声学建模目的，并处理在典型的室内声学场景中直接暴露或被薄表面覆盖物覆盖的多孔吸声器。为了充分表示边界的扩展反应，对所有传播介质中的声波传播进行了模拟和显式耦合。基于有理逼近和ADE方法的结合，通过对EFM进行变换，构建了多孔介质域的时域控制方程。这种通用公式不仅可以实现材料模型的灵活性，而且可以实现一致且高效的数值离散。这项工作的第二个贡献是开发了精确的黎曼求解器，它是一种受物理启发的数值通量公式，用于具有跳跃不连续性的传播介质之间的精确耦合。本文的其余部分组织如下。第 2 节以统一双曲线形式提出了声音在空气和多孔材料中传播的控制方程。第 3 节描述了数值方案。在 $^ { 4 , }$ 节中执行数值验证和应用。结论性意见见第 5 节。

The main focus of this work is to present a time-domain formulation for extended reacting boundary of porous materials within the framework of the time-domain discontinuous Galerkin method. The proposed formulation serves room acoustic modeling purposes and handles porous absorbers that are either directly exposed or covered by thin surface coverings as in typical room acoustic scenarios. To fully represent the extended reaction of boundaries, acoustic wave propagation in all propagation media are simulated and coupled explicitly. The time-domain governing equations for the porous media domain are constructed by transforming EFMs based on the combination of rational approximations and the ADE method. This general formulation enables not only flexibilities in material models but also consistent and efficient numerical discretizations. The second contribution of this work is that an exact Riemann solver, which is a physically inspired numerical flux formulation, is developed for precise coupling between the propagation media with jump discontinuities. The rest of this paper is organized as follows. Section 2 presents the governing equations of sound propagation in air and porous materials in a unified hyperbolic form. Section 3 describes the numerical schemes. In Section $^ { 4 , }$ numerical verifications and applications are performed. Concluding remarks are given in Section 5.

## 2. 时域公式

## 2. Time-domain formulations

为了清楚起见，由两种均匀传播介质（空气和多孔材料）及其各自的物理控制方程和属性组成的有界域 $\varOmega ,$ 被认为是说明本工作中的公式。该公式可以直接扩展到涉及不同属性的多种传播介质的问题。

For the purpose of clarity, a bounded domain $\varOmega ,$ composed of two homogeneous propagation media (the air and the porous material) with their respective physical governing equations and properties, is considered to illustrate the formulations in this work. This formulation can be extended to problems involving multiple propagation media of different properties straightforwardly.

## 2.1. 空气域

## 2.1. Air domain

声波在静止空气子域 $\varOmega _ { a }$ 中的传播可以用线性声学方程来描述

Acoustic wave propagation in the motionless air subdomain $\varOmega _ { a }$ can be described by linear acoustic equations

$$
\begin{array}{r} \frac {\partial \mathbf {v}}{\partial t} + \frac {1}{\rho_ {a}} \nabla p = \mathbf {0}, \\ \frac {\partial p}{\partial t} + \rho_ {a} c _ {a} ^ {2} \nabla \cdot \mathbf {v} = 0, \end{array}\tag{1}
$$

(2) 

其中 $\mathbf { v } ( \mathbf { x } , t )$ 是分量为 $\{ v _ { x } , v _ { y } , v _ { z } \}$ 的粒子速度矢量， ??(??, ??) 是声压， ??表示空间位置，$\rho _ { a } = 1 . 2 ~ \mathrm { k g } / \mathrm { m } ^ { 3 }$ 是空气的恒定密度，$c _ { a } = 3 4 3 ~ \mathrm { m / s }$ 是恒定的声速。

where $\mathbf { v } ( \mathbf { x } , t )$ is the particle velocity vector with components $\{ v _ { x } , v _ { y } , v _ { z } \}$ , ??(??, ??) is the sound pressure, ?? denotes the spatial position, $\rho _ { a } = 1 . 2 ~ \mathrm { k g } / \mathrm { m } ^ { 3 }$ is the constant density of air and $c _ { a } = 3 4 3 ~ \mathrm { m / s }$ is the constant speed of sound.

## 2.2. 多孔材料域

## 2.2. Porous material domain

控制多孔材料子域 $\varOmega _ { m }$ 内的声音传播的 EFM 在频域中表示为（假设 $\mathrm { e } ^ { \mathrm { i } \omega t }$ 时间约定）

The EFMs governing sound propagation inside the porous material subdomain $\varOmega _ { m }$ are expressed in the frequency-domain as (assuming $\mathrm { e } ^ { \mathrm { i } \omega t }$ time convention)

$$
\begin{array}{r} \mathrm{i} \omega \rho_ {\mathrm{ef}} (\omega) \hat {\mathbf {v}} + \nabla \hat {p} = \mathbf {0}, \\ \mathrm{i} \omega \mathcal {C} _ {\mathrm{ef}} (\omega) \hat {p} + \nabla \cdot \hat {\mathbf {v}} = 0, \end{array}\tag{3a}
$$

(3b) 

其中帽子符号 (̂⋅) 标记频域变量，$\rho _ { \mathrm { e f } } ( \omega )$ 是有效密度，$c _ { \mathrm { e f } } ( \omega )$ 是多孔介质的有效压缩性（即有效体积模量的倒数）。为了获得时域控制方程，我们首先使用具有实极点的有理函数来近似复值 $\rho _ { \mathrm { e f } } ( \omega )$ 和 $c _ { \mathrm { e f } } ( \omega )$

where the hat notation (̂⋅) labels frequency-domain variables, $\rho _ { \mathrm { e f } } ( \omega )$ is the effective density, $c _ { \mathrm { e f } } ( \omega )$ is the effective compressibility (i.e. the inverse of effective bulk modulus) of the porous medium. To obtain time-domain governing equations, we first approximate the complex-valued $\rho _ { \mathrm { e f } } ( \omega )$ and $c _ { \mathrm { e f } } ( \omega )$ using rational functions with real poles as

$$
\begin{array}{r} \rho_ {\mathrm{ef}} (\omega) \approx \rho_ {m} + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} \frac {B _ {\rho k}}{\zeta_ {\rho k} + \mathrm{i} \omega}, \\ \mathcal {C} _ {\mathrm{ef}} (\omega) \approx \mathcal {C} _ {m} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} \frac {B _ {C k}}{\zeta_ {C k} + \mathrm{i} \omega}, \end{array}\tag{4a}
$$

(4b) 

其中 $[ B _ { \rho k } , B _ { C k } ] \in \mathbb { R }$ 和 $[ \zeta _ { \rho k } , \zeta _ { C k } ] \in \mathbb { R } ^ { + }$ 分别是拟合权重和极点。 $\rho _ { m }$ 是频率接近无穷大时有效密度的渐近值，而 $c _ { m }$ 表示有效压缩性的高频渐近值。由于两者都是常数，因此材料 $c _ { m }$ 中的声速渐近值是一个等于 $1 / \sqrt { \rho _ { m } C _ { m } } .$ 的常数。关于 EFM 的最新数值研究 $[ 3 6 , 3 9 , 5 2 ]$ 和分析 [53] 表明，具有实极点的有理函数足以捕捉传统多孔材料的耗散性质。为了获得稳定的解，所有极点必须为正，如[36]中所证明的。在这项工作中，矢量拟合（VF）算法[54]可以明确地施加这种稳定性条件，由于其高精度和高效率，用于确定拟合参数。通过代入等式。 (4) 代入等式。 (3) 并将部分分数分解应用于项 i\omega $\rho _ { \mathrm { e f } }$ 和我?? $c _ { \mathrm { e f } }$ 其中，我们得到

where $[ B _ { \rho k } , B _ { C k } ] \in \mathbb { R }$ and $[ \zeta _ { \rho k } , \zeta _ { C k } ] \in \mathbb { R } ^ { + }$ are fitting weights and poles respectively. $\rho _ { m }$ is the asymptotic value of effective density as the frequency approaches infinity, whereas $c _ { m }$ denotes the high-frequency asymptotic value of the effective compressibility. As both of them are constant, the asymptotic value of the sound speed in the material $c _ { m }$ is a constant that is equal to $1 / \sqrt { \rho _ { m } C _ { m } } .$ . Latest numerical studies $[ 3 6 , 3 9 , 5 2 ]$ and analysis [53] on EFM have demonstrated that rational functions with real poles are sufficient to capture the dissipative nature of conventional porous materials. In order to obtain stable solutions, all poles must be positive as proved in [36]. In this work, a vector fitting (VF) algorithm [54], which can impose this stability condition explicitly, is used for determining the fitting parameters due to its high accuracy and efficiency. By substituting Eqs. (4) into Eqs. (3) and applying the partial fraction decomposition to terms i\omega $\rho _ { \mathrm { e f } }$ and i\omega $c _ { \mathrm { e f } }$ therein, we get

$$
\mathrm{i} \omega \rho_ {m} \hat {\mathbf {v}} + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} \big (B _ {\rho k} - \frac {B _ {\rho k} \zeta_ {\rho k}}{\zeta_ {\rho k} + \mathrm{i} \omega} \big) \hat {\mathbf {v}} + \nabla \hat {p} = \mathbf {0},\tag{5a}
$$

$$
\mathrm{i} \omega \mathcal {C} _ {m} \hat {p} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} \big (B _ {C k} - \frac {B _ {C k} \zeta_ {C k}}{\zeta_ {C k} + \mathrm{i} \omega} \big) \hat {p} + \nabla \cdot \hat {\mathbf {v}} = 0.\tag{5b}
$$

然后，应用傅里叶逆变换和辅助微分方程（ADE）方法[35,55]得到

Then, applying the inverse Fourier transform and the auxiliary differential equations (ADE) method [35,55] results in

$$
\begin{array}{r} \rho_ {m} \frac {\partial \mathbf {v}}{\partial t} + \nabla p + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \mathbf {v} - \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \boldsymbol {\phi} _ {\rho k} = \mathbf {0}, \\ \frac {1}{\rho_ {m} c _ {m} ^ {2}} \frac {\partial p}{\partial t} + \nabla \cdot \mathbf {v} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} p - \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \zeta_ {C k} \boldsymbol {\phi} _ {C k} = 0, \end{array}\tag{6a}
$$

(6b) 

其中 $\boldsymbol { \phi } _ { \rho k } = [ \phi _ { \rho k } ^ { x } , \phi _ { \rho k } ^ { y } , \phi _ { \rho k } ^ { z } ] ^ { \mathrm { T } }$ 和 $\phi _ { C k }$ 是所谓的累加器或辅助变量，它们对应于与 ?? 的分量相关的卷积积分。和 $p ,$ 分别，例如，

where $\boldsymbol { \phi } _ { \rho k } = [ \phi _ { \rho k } ^ { x } , \phi _ { \rho k } ^ { y } , \phi _ { \rho k } ^ { z } ] ^ { \mathrm { T } }$ and $\phi _ { C k }$ are the so-called accumulators or auxiliary variables that correspond to the convolution integral associated with the components of ?? and $p ,$ respectively, e.g.,

$$
\phi_ {\rho k} (\mathbf {x}, t) = \int_ {0} ^ {t} \mathbf {v} (\mathbf {x}, \tau) \mathrm{e} ^ {- \zeta_ {\rho k} (t - \tau)} \mathrm{d} \tau .
$$

它们由瞬态常微分方程控制

They are governed by time-dependent ordinary differential equations

$$
\begin{array}{r l} & {\frac {\partial \pmb {\phi} _ {\rho k}}{\partial t} + \zeta_ {\rho k} \pmb {\phi} _ {\rho k} = \mathbf {v}, \quad \forall k \in [ 1, \mathcal {N} _ {\rho} ],} \\ & {\frac {\partial \phi_ {C k}}{\partial t} + \zeta_ {C k} \phi_ {C k} = p, \quad \forall k \in [ 1, \mathcal {N} _ {C} ],} \end{array}\tag{7a}
$$

(7b) 

初始条件为零。等式。 (6)，加上等式。 (7) 形成多孔材料中波传播的增广时域控制方程组。如方程式所示。 (4)，频率相关属性的影响现在通过频率无关渐近值 $\{ \rho _ { m } , c _ { m } \}$ 和频率相关辅助变量 $\{ \phi _ { \rho k } , \phi _ { C k } \}$ 的叠加来体现。与先前关于 ER 边界公式的相关工作相比 [37]，所提出的时域公式完全排除了辅助微分方程中主要声学变量的空间导数。 (7) 由于方程中部分分数分解的应用。 (5) 遵循[36,39]。此外，由于系统中不存在辅助变量的空间导数，因此与空间导数近似相关的部分计算成本不会随着辅助变量的数量而增加。正如将在下面的 3.1 节中看到的，该公式为空间离散化方案的推导提供了宝贵的便利，并且可以应用一致的数值处理来离散空气和多孔材料域。需要注意的是，辅助变量需要以与材料域中的主要声学变量相同的方式定义和时间积分，这导致存储器存储和时间积分成本不可避免地增加。

with zero initial conditions. Eqs. (6), coupled with Eqs. (7), form the augmented system of time-domain governing equations for wave propagation in porous materials. As indicated by Eqs. (4), the effects of frequency-dependent properties are now manifested by a superimposition of the frequency-independent asymptotic values $\{ \rho _ { m } , c _ { m } \}$ and the frequency-dependent auxiliary variables $\{ \phi _ { \rho k } , \phi _ { C k } \}$ . Compared to the relevant prior work on the ER boundary formulation [37], the proposed time-domain formulation completely excludes spatial derivatives of primary acoustic variables from the auxiliary differential Eqs. (7) thanks to application of the partial fractional decomposition in Eqs. (5) following [36,39]. Furthermore, since there is no spatial derivative of the auxiliary variables in the system, the part of the computational cost related to the spatial derivative approximation does not increase with the number of the auxiliary variables. As will be seen in the following Section 3.1, this formulation offers valuable convenience in the derivation of the spatial discretization scheme, and a consistent numerical treatment can be applied to discretize both the air and the porous material domains. It should be noted that the auxiliary variables need to be defined and time integrated in the same way as the primary acoustic variables in the material domain, resulting in an unavoidable increase in terms of the memory storage and the time integration cost.

## 2.3. 界面条件

## 2.3. Interface conditions

所考虑的空气（由下标 ?? 表示）和多孔材料（由 ?? 表示）之间的薄覆盖物通过包含压力跃变和连续法向速度的界面条件进行建模[56]，即，

The considered thin covering between the air (denoted by the subscript ??) and the porous material (denoted by ??) is modeled by an interface condition containing a pressure jump and continuous normal velocity [56], i.e.,

$$
\begin{array}{r} \mathbf {v} _ {a} \cdot \mathbf {n} _ {a} = - \mathbf {v} _ {m} \cdot \mathbf {n} _ {m}, \\ p _ {a} - p _ {m} = Z _ {t} \mathbf {v} _ {a} \cdot \mathbf {n} _ {a}, \end{array}\tag{8a}
$$

(8b) 

其中 ${ \mathbf { n } } _ { a }$ 和 $\mathbf { n } _ { m }$ 是满足 $\mathbf { n } _ { a } = - \mathbf { n } _ { m }$ 的每个子域边界表面处的单位向外法向量。薄覆盖物的传输阻抗表示为 $Z _ { t }$ 。当多孔材料与空气域直接接触时，它变为零。值得一提的是，传递阻抗模型直接考虑了覆盖层厚度对吸声的影响。由于覆盖物的厚度通常远小于声波波长，因此可以忽略覆盖物的边缘衍射效应。这种压力跃变界面模型已成功应用于模拟声音在渗透性薄膜[44-46]、穿孔板[39]和微穿孔板[47-48]之间的传播。

where ${ \mathbf { n } } _ { a }$ and $\mathbf { n } _ { m }$ are the unit outward normal vector at the boundary surface of each subdomain satisfying $\mathbf { n } _ { a } = - \mathbf { n } _ { m }$ . The transfer impedance of the thin covering is denoted as $Z _ { t }$ . It becomes zero when the porous material is in direct contact with the air domain. It is worth mentioning that the effect of covering thickness on sound absorption is taken into account by the transfer impedance model directly. Since the thickness of the covering is typically much smaller than the acoustic wavelength, the edge diffraction effect of the covering is neglected. This pressure jump interface model has been successfully applied to simulate sound transmission across permeable thin membranes [44–46], perforated plates [39] and microperforated panels [47 48]

## 3. 数值格式

## 3. Numerical schemes

## 3.1. 使用 DG 方法的空间离散

## 3.1. Spatial discretization with the DG method

为了离散空间导数算子，我们首先将两个子域中的主要声学状态变量的控制方程（空气的方程（1）和多孔材料的方程（6））重写为一般的一阶双曲形式，如下所示

To discretize the spatial derivative operators, we first rewrite the governing equations for the primary acoustic state variables in both subdomains (Eqs. (1) for the air and Eqs. (6) for the porous material), into a general first-order hyperbolic form as follows

$$
\frac {\partial \mathbf {q}}{\partial t} + \mathbf {A} _ {x} \frac {\partial \mathbf {q}}{\partial x} + \mathbf {A} _ {y} \frac {\partial \mathbf {q}}{\partial y} + \mathbf {A} _ {z} \frac {\partial \mathbf {q}}{\partial z} + \mathbf {D q} = \mathbf {g},\tag{9}
$$

其中 $\mathbf { q } ( \mathbf { x } , t ) = [ v _ { x } , v _ { y } , v _ { z } , p ] ^ { \mathrm { T } }$ 表示主要声学变量向量。通量雅可比矩阵 $\mathbf { A } _ { i } ~ ( j \in \{ x , y , z \} )$

where $\mathbf { q } ( \mathbf { x } , t ) = [ v _ { x } , v _ { y } , v _ { z } , p ] ^ { \mathrm { T } }$ denotes the primary acoustic variable vector. The flux Jacobian matrices $\mathbf { A } _ { i } ~ ( j \in \{ x , y , z \} )$

$$
\mathbf {A} _ {j} = \left[ \begin{array}{c c c c} 0 & 0 & 0 & \frac {\delta_ {x j}}{\rho} \\ 0 & 0 & 0 & \frac {\delta_ {y j}}{\rho} \\ 0 & 0 & 0 & \frac {\delta_ {z j}}{\rho} \\ \rho c ^ {2} \delta_ {x j} & \rho c ^ {2} \delta_ {y j} & \rho c ^ {2} \delta_ {z j} & 0 \end{array} \right],
$$

(10) 

具有与频率无关的常数项，其中密度 ??和声速？？取每个子域相应的常数对值，即空气的$\{ \rho , c \} = \{ \rho _ { a } , c _ { a } \}$，多孔材料的$\{ \rho , c \} = \{ \rho _ { m } , c _ { m } \}$，$\delta _ { i j }$表示克罗内克δ函数。松弛矩阵 ?? 和右侧类似源项 ??完全负责传播介质的频率（in）依赖性。具体来说，对于具有频率无关介质属性的空气域，两者都 ??和 ？？为空。对于多孔材料域，松弛矩阵 ??和术语？是与频率相关的辅助变量 $\{ \phi _ { \rho k } , \phi _ { C k } \}$ 的函数，如下所示

have frequency-independent constant entries, where the density ?? and speed of sound ?? takes the corresponding constant pair of values of each subdomain, i.e., $\{ \rho , c \} = \{ \rho _ { a } , c _ { a } \}$ for the air while $\{ \rho , c \} = \{ \rho _ { m } , c _ { m } \}$ for the porous material, and $\delta _ { i j }$ denotes the Kronecker delta function. The relaxation matrix ??, and the right-hand side source-like term ?? are fully responsible for the frequency-(in)dependency of the propagation medium. Specifically, for the air domain with frequency-independent medium properties, both ?? and ?? are null. For the porous material domain, the relaxation matrix ?? and term ?? are functions of the frequency-dependent auxiliary variables $\{ \phi _ { \rho k } , \phi _ { C k } \}$ as

$$
\mathbf {D} = \left[ \begin{array}{c c c c} \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 & 0 & 0 \\ 0 & \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 & 0 \\ 0 & 0 & \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 \\ 0 & 0 & 0 & \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \end{array} \right], \quad \mathbf {g} = \left[ \begin{array}{c} \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {x} \\ \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {y} \\ \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {z} \\ \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \zeta_ {C k} \phi_ {C k} \end{array} \right].\tag{11}
$$

由于频率相关项 ??和 ？？与通量雅可比矩阵隔离，待空间离散的空气-材料耦合系统被转化为具有分段恒定材料属性的系统。

Since the frequency-dependent terms ?? and ?? are isolated from the flux Jacobian matrices, the air–material coupled system to be spatially discretized is transformed into one with piece-wise constant material properties.

求解方程。 (9) 用DG方法，物理域??被划分为一组不重叠的元素 $\varOmega ^ { e } .$ 。遵循节点 DG 公式 [57,58]，在每个元素 $\varOmega ^ { e }$ 中，未知解 $\mathbf { q } _ { h } ^ { e } ( { \bf x } , t )$ 的局部分段多项式近似表示为：

To solve Eq. (9) with the DG method, the physical domain ?? is partitioned into a set of non-overlapping elements $\varOmega ^ { e } .$ . Following the nodal DG formulation [57,58], in each element $\varOmega ^ { e }$ , a local piecewise polynomial approximation of the unknown solution $\mathbf { q } _ { h } ^ { e } ( { \bf x } , t )$ is expressed by:

$$
\mathbf {q} ^ {e} (\mathbf {x}, t) \approx \mathbf {q} _ {h} ^ {e} (\mathbf {x}, t) = \sum_ {i = 1} ^ {N _ {p}} \mathbf {q} _ {h} ^ {e} (\mathbf {x} _ {i} ^ {e}, t) l _ {i} ^ {e} (\mathbf {x}),\tag{12}
$$

其中下标 ℎ 表示数值近似， $\mathbf { q } _ { h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) = [ v _ { x h } ^ { e } , v _ { \nu h } ^ { e } , v _ { z h } ^ { e } , p _ { h } ^ { e } ] ^ { \mathrm { T } }$ 是位置处的未知节点值 $\mathbf { x } _ { i } ^ { e } , ~ l _ { i } ^ { e } ( \mathbf { x } )$ 是阶数 ?? 的多维拉格朗日多项式基满足 $l _ { i } ^ { e } ( { \bf x } _ { i } ^ { e } ) = \delta _ { i j } . ~ N _ { p }$ 是单个元素内局部基函数（自由度）的数量，对于单纯形元素等于 $( N + d ) ! / ( N ! d ! )$，其中 ??是维度。基函数 $l _ { i } ^ { e } ( { \bf x } )$ 由节点分布 ${ \bf x } _ { i } ^ { e } ,$ 确定，在本研究中，Legendre–Gauss–Lobatto (LGL) 求积点用于一维问题，??优化节点分布 [57] 用于多维元素。相同的局部多项式近似应用于多孔材料域 $\varOmega _ { m }$ 中的辅助变量 $\phi _ { \rho k }$ 和 $\phi _ { C k }$

where the subscript ℎ denotes the numerical approximation, $\mathbf { q } _ { h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) = [ v _ { x h } ^ { e } , v _ { \nu h } ^ { e } , v _ { z h } ^ { e } , p _ { h } ^ { e } ] ^ { \mathrm { T } }$ are the unknown nodal values at locations $\mathbf { x } _ { i } ^ { e } , ~ l _ { i } ^ { e } ( \mathbf { x } )$ is the multi-dimensional Lagrange polynomial basis of order ?? satisfying $l _ { i } ^ { e } ( { \bf x } _ { i } ^ { e } ) = \delta _ { i j } . ~ N _ { p }$ is the number of local basis functions (the degree of freedom) inside a single element and is equal to $( N + d ) ! / ( N ! d ! )$ for simplex elements, where ?? is the dimensionality. The basis functions $l _ { i } ^ { e } ( { \bf x } )$ are determined by the nodal distribution ${ \bf x } _ { i } ^ { e } ,$ , and in this study, the Legendre–Gauss–Lobatto (LGL) quadrature points are used for 1D problems and the ??-optimized nodal distribution [57] are used for multi-dimensional elements. The same local polynomial approximations are applied to the auxiliary variables $\phi _ { \rho k }$ and $\phi _ { C k }$ in the porous material domain $\varOmega _ { m }$

变分公式是通过乘以方程得到的。 (9) 使用局部测试函数 $l _ { i } ^ { e } ( { \bf x } )$ 并按部分积分两次，得到

The variational formulation is obtained by multiplying Eq. (9) with a local test function $l _ { i } ^ { e } ( { \bf x } )$ and integration by parts twice, yielding

$$
\int_ {\Omega^ {e}} l _ {i} ^ {e} \left(\frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial t} + \mathbf {A} _ {x} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial x} + \mathbf {A} _ {y} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial y} + \mathbf {A} _ {z} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial z} + \mathbf {D} \mathbf {q} _ {h} ^ {e} - \mathbf {g}\right) d \mathbf {x} = \oint_ {\partial \Omega^ {e}} l _ {i} ^ {e} \left(\mathbf {A} _ {n} ^ {e} \mathbf {q} _ {h} ^ {e} - \mathbf {F} ^ {e} (\mathbf {q} _ {h} ^ {e}, \mathbf {q} _ {h} ^ {e +})\right) d \mathbf {x},\tag{13}
$$

其中 $\mathbf { A } _ { n } ^ { e } : = \mathbf { A } _ { x } n _ { x } ^ { e } + \mathbf { A } _ { y } n _ { \nu } ^ { e } + \mathbf { A } _ { z } n _ { z } ^ { e }$ 是沿单元界面 ????<sup>??</sup> 的向外法向矢量 $\mathbf { n } _ { e } = [ n _ { x } ^ { e } , n _ { v } ^ { e } , n _ { z } ^ { e } ]$ 的法向通量矩阵。 ${ \bf F } ^ { e } ( { \bf q } _ { h } ^ { e } , { \bf q } _ { h } ^ { e + } )$ ，即所谓的跨单元表面的数值通量 $\partial \varOmega ^ { e } ,$ ，是局部解值 $\mathbf { q } _ { h } ^ { e }$ 和来自界面另一侧的相邻单元的解值 ${ \bf q } _ { h } ^ { e + }$ 的函数。众所周知。数值通量是 DG 方法稳定性和准确性的重要因素[59]。除了连接相邻的内部元素之外，数值通量还可以弱地施加边界条件并保证公式的稳定性。通常，使用三种类型的通量：中心通量、Lax-Friedrich 通量和黎曼求解器（即逆风通量或 Godunov 通量）。中心通量是最简单的方案，仅取界面两侧的平均值，并且本质上是非耗散的。 Lax-Friedrich 通量通过直接添加基于最大波传播速度的耗散项来稳定 DG 方法。由于缺乏适当的耗散来消除杂散模式，它们都容易受到长期不稳定问题的影响[60,61]。相比之下，通过考虑基础物理并适当消除杂散模式，基于黎曼求解器的迎风通量具有最高的耗散和色散特性[62]。因此，在本作品中使用了它。在下一节中，导出了与空气和多孔材料之间的跳跃界面对齐的单元表面的数值通量公式，该公式可以直接应用于内部单元界面的简化情况以及不存在薄覆盖物的情况。

where $\mathbf { A } _ { n } ^ { e } : = \mathbf { A } _ { x } n _ { x } ^ { e } + \mathbf { A } _ { y } n _ { \nu } ^ { e } + \mathbf { A } _ { z } n _ { z } ^ { e }$ is the normal flux matrix along the outward normal vector $\mathbf { n } _ { e } = [ n _ { x } ^ { e } , n _ { v } ^ { e } , n _ { z } ^ { e } ]$ of the element interface ????<sup>??</sup>. ${ \bf F } ^ { e } ( { \bf q } _ { h } ^ { e } , { \bf q } _ { h } ^ { e + } )$ , the so-called numerical flux across element surface $\partial \varOmega ^ { e } ,$ , is a function of both the local solution value $\mathbf { q } _ { h } ^ { e }$ and the solution value ${ \bf q } _ { h } ^ { e + }$ of the neighboring element from the other side of the interface surface. As is known. the numerical flux is a paramount factor on the stability and accuracy of the DG method [59]. Apart from linking neighboring interior elements, the numerical flux also serves to impose boundary conditions weakly and to guarantee the stability of the formulation. Typically, three types of fluxes, central flux, Lax–Friedrich flux, and Riemann solver (i.e., upwind flux or Godunov flux), are used. The central flux is the simplest scheme that just takes the average values on both sides of an interface and is non-dissipative by nature. The Lax-Friedrich flux stabilizes the DG method by straightforwardly adding a dissipative term based on the maximum wave propagation speed. Both of them are vulnerable to a long-time instability issue due to the lack of proper dissipation to eliminate spurious modes [60,61]. By contrast, the upwind flux based on the Riemann solver has supreme dissipation and dispersion properties [62] by considering underlying physics and eliminating spurious modes properly. Therefore, it is used in this work. In the following section, the numerical flux formulation is derived for the element surface aligned with the jump interface between the air and the porous material, which can be straightforwardly applied to the simplified case of interior element interfaces and to the scenario where the thin covering is absent.

## 3.2. 空气-材料界面的黎曼求解器

## 3.2. Riemann solver across air–material interface

迎风数值通量是基于黎曼问题的解构建的，该问题考虑了具有频率无关特性的两种均匀介质之间的界面，如恒定通量雅可比矩阵所表示的。遵循符号约定[57,63,64]，上标“-”和“+”用于强调这两种媒体内的值。剩下的，?? $: = { \bf n } ^ { - } = [ n _ { x } , n _ { y } , n _ { z } ]$ 表示向外单位法线向量。恒定介质属性由 $\{ \rho ^ { - } , c ^ { - } \}$ 沿 ?? 向内方向表示。和 $\{ \rho ^ { + } , c ^ { + } \}$ 在 ?? 的向外方向。从数学上来说，黎曼问题就是解决方程（1）。 (9) 对于给定的具有不连续初始条件的分段常数介质

The upwind numerical flux is constructed based on the solutions of the Riemann problem, which considers the interface between two homogeneous media with frequency-independent properties as represented by the constant flux Jacobian matrices. Following the notation convention [57,63,64], the superscripts ‘‘ − ’’ and ‘‘ + ’’ are used to emphasize the values inside these two media. In the remainder, ?? $: = { \bf n } ^ { - } = [ n _ { x } , n _ { y } , n _ { z } ]$ denotes the outward unit normal vector. The constant media properties are denoted by $\{ \rho ^ { - } , c ^ { - } \}$ in the inward direction of ?? and $\{ \rho ^ { + } , c ^ { + } \}$ in the outward direction of ??. Mathematically, the Riemann problem is to solve Eqs. (9) for the given piecewise constant medium with the discontinuous initial condition

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/29d3e3a6522a5bbb83c8671c96539b0694d8979ede2eece59f643f95b5883fec.jpg)



图 1. 黎曼问题的 Rankine-Hugoniot 跳跃条件图示。具有不同波传播速度和方向的特征由箭头线表示。

Fig. 1. Illustration of the Rankine-Hugoniot jump conditions for the Riemann problem. Characteristics with distinct wave propagation speeds and directions are denoted by the arrow lines.


$$
\mathbf {q} _ {0} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {q} ^ {-} & \text {if} \quad \mathbf {n} \cdot (\mathbf {x} - \mathbf {x} _ {0}) <   0 \\ \mathbf {q} ^ {+} & \text {if} \quad \mathbf {n} \cdot (\mathbf {x} - \mathbf {x} _ {0}) > 0 \end{array} \right.
$$

其中 $\mathbf { x } _ { 0 }$ 位于接口上。为了解决黎曼问题，首先通过通量矩阵 $\mathbf { A } _ { n }$ 沿 ?? 的特征分解来确定系统的特性。

where $\mathbf { x } _ { 0 }$ lies on the interface. To solve the Riemann problem, the characteristics of the system are first determined by the eigendecomposition of the flux matrix $\mathbf { A } _ { n }$ along ??

$$
\mathbf {A} _ {n} := \mathbf {A} _ {x} n _ {x} + \mathbf {A} _ {y} n _ {y} + \mathbf {A} _ {z} n _ {z} = \mathbf {R} \boldsymbol {\Lambda} \mathbf {R} ^ {- 1},\tag{14}
$$

特征矩阵读取的位置

where the eigenmatrix reads

$$
\mathbf {R} = \frac {1}{2} \left[ \begin{array}{c c c c} - n _ {x} & - 2 n _ {y} & - 2 n _ {z} & n _ {x} \\ - n _ {y} & 2 n _ {x} & 0 & n _ {y} \\ - n _ {z} & 0 & 2 n _ {x} & n _ {z} \\ \rho c & 0 & 0 & \rho c \end{array} \right],\tag{15}
$$

Λ 是特征值矩阵，其对角线条目由 $\lambda _ { 1 } = - c , \lambda _ { 2 } = \lambda _ { 3 } = 0 , \lambda _ { 4 } = c ,$ 给出，表示特征波速。 Rankine-Hugoniot (RH) 跳跃条件 [57,63,64]

and Λ is the eigenvalue matrix with diagonal entries given by $\lambda _ { 1 } = - c , \lambda _ { 2 } = \lambda _ { 3 } = 0 , \lambda _ { 4 } = c ,$ representing the characteristic wave speed. The Rankine–Hugoniot (RH) jump condition [57,63,64]

$$
- \lambda_ {i} (\mathbf {q} ^ {\text { neg }} - \mathbf {q} ^ {\text { pos }}) + \mathbf {A} _ {n} (\mathbf {q} ^ {\text { neg }} - \mathbf {q} ^ {\text { pos }}) = \mathbf {0}\tag{16}
$$

速度为 $\lambda _ { i } ( i \in \{ 1 , 2 , 3 , 4 \} )$ 的每个特征波都成立，其中 $\mathbf { q } ^ { \mathrm { n e g } }$ 是特征波负法线方向上的状态向量，${ \bf q } ^ { \mathrm { p o s } }$ 是正法线方向上的状态向量。矩阵 $\mathbf { A } _ { n }$ 使用 $\lambda _ { i }$ 特征波行进区域中的相应材料属性值 $\{ \rho , c \}$ 进行评估，如图 1 所示，该区域由上标 $^ { 6 6 } - { } ^ { 5 5 }$ 和 $^ { 6 \prime } + \boldsymbol { ^ { \prime \prime } }$ 强调。上述RH跳变条件(16)可以进一步例证为

holds across each characteristic wave of speed $\lambda _ { i } ( i \in \{ 1 , 2 , 3 , 4 \} )$ , where $\mathbf { q } ^ { \mathrm { n e g } }$ is the state vector in the negative normal direction across the characteristic wave and ${ \bf q } ^ { \mathrm { p o s } }$ is the state vector in the positive normal direction. Matrix $\mathbf { A } _ { n }$ is evaluated using the corresponding material property value $\{ \rho , c \}$ in the region where the $\lambda _ { i }$ characteristic wave travels as shown in Fig. 1, which is emphasized by the superscripts $^ { 6 6 } - { } ^ { 5 5 }$ and $^ { 6 \prime } + \boldsymbol { ^ { \prime \prime } }$ . The above RH jump condition (16) can be further exemplified as

$$
\begin{array}{c} c ^ {-} (\mathbf {q} ^ {-} - \mathbf {q} ^ {a}) + \mathbf {A} _ {n} ^ {-} (\mathbf {q} ^ {-} - \mathbf {q} ^ {a}) = \mathbf {0}, \\ \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} - \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {0}, \\ - c ^ {+} (\mathbf {q} ^ {b} - \mathbf {q} ^ {+}) + \mathbf {A} _ {n} ^ {+} (\mathbf {q} ^ {b} - \mathbf {q} ^ {+}) = \mathbf {0}. \end{array}\tag{17a}
$$

(17b) 

(17c) 

两个未知的中间状态 $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ 是所考虑系统的实际黎曼解。基于 $\mathbf { A } _ { n } ,$ 特征向量的数学定义，RH 条件等价地表明状态向量中的跳跃是对应侧特征向量的线性组合 [63,65]：

Two unknown intermediate states $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ are the actual Riemann solutions for the considered system. Based on the mathematical definition of eigenvectors of $\mathbf { A } _ { n } ,$ the RH condition equivalently states that the jumps in state vector are a linear combination of that corresponding side’s eigenvectors [63,65]:

$$
\begin{array}{r} \mathbf {q} ^ {-} - \mathbf {q} ^ {a} = \alpha_ {1} \mathbf {r} _ {1} ^ {-}, \\ \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} - \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {0}, \\ \mathbf {q} ^ {b} - \mathbf {q} ^ {+} = \alpha_ {4} \mathbf {r} _ {4} ^ {+}, \end{array}\tag{18a}
$$

(18b) 

(18c) 

其中$\mathbf { r } _ { i }$表示$\mathbf { R } ,$的第列，${ \pmb { \alpha } } = [ \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } , \alpha _ { 4 } ]$表示待求解的特征系数，对应于速度的特征波$\lambda _ { i } ~ ( i \in \{ 1 , 2 , 3 , 4 \} )$)。请注意，零速特征波对迎风数值通量没有贡献，因此不考虑 $\alpha _ { 2 }$ 和 $\alpha _ { 3 }$。

where $\mathbf { r } _ { i }$ denotes the ??th column of $\mathbf { R } ,$ and ${ \pmb { \alpha } } = [ \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } , \alpha _ { 4 } ]$ denotes the characteristic coefficients to be solved, corresponding to the characteristic wave of speed $\lambda _ { i } ~ ( i \in \{ 1 , 2 , 3 , 4 \} )$ ). Note that characteristic waves with zero speed do not contribute to the upwind numerical flux, therefore $\alpha _ { 2 }$ and $\alpha _ { 3 }$ are not considered.

从数学上讲，RH跳跃条件（18）的系统是欠定的，需要额外的条件才能唯一求解。从物理角度来看，界面两侧之间的连接应该通过在中间状态 $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ 上施加适当的物理边界条件来建立。求解方程。 (18) 符合式(18)的接口条件(8) 得到特征系数 $\{ \alpha _ { 1 } , \alpha _ { 4 } \}$ 的解为

Mathematically speaking, the system of the RH jump condition (18) is underdetermined and needs extra conditions in order to be solved uniquely. From the physical perspective, connections between the two sides across the interface are supposed to be established by imposing appropriate physical boundary conditions over the intermediate states $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ . Solving Eqs. (18) with the interface condition of Eq. (8) yields the solutions of characteristic coefficient $\{ \alpha _ { 1 } , \alpha _ { 4 } \}$ as

$$
\left[ \begin{array}{c} \alpha_ {1} \\ \alpha_ {4} \end{array} \right] = \frac {2}{Z ^ {-} + Z _ {t} + Z ^ {+}} \left[ \begin{array}{c} p ^ {-} - p ^ {+} - Z ^ {+} v _ {n} ^ {+} - (Z _ {t} + Z ^ {+}) v _ {n} ^ {-} \\ p ^ {-} - p ^ {+} + Z ^ {-} v _ {n} ^ {-} + (Z _ {t} + Z ^ {-}) v _ {n} ^ {+} \end{array} \right],\tag{19}
$$

其中引入了两种均匀介质 $Z ^ { - } = \rho ^ { - } c ^ { - }$ 和 $Z ^ { + } = \rho ^ { + } c ^ { + }$ 的特性阻抗以及 $v _ { n } ^ { + } = \mathbf { v } ^ { + } \cdot \mathbf { n } ^ { + } =$ $- \mathbf { v } ^ { + } \cdot \mathbf { n } , v _ { n } ^ { - } = \mathbf { v } ^ { - } \cdot \mathbf { n }$ 。代入计算出的 ??从方程式(19) 代入等式。 (18) 得出黎曼解 $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ }。对于 ?? 向内方向的单元，迎风数值通量 ??沿着 ？？读

where the characteristic impedance of the two homogeneous media $Z ^ { - } = \rho ^ { - } c ^ { - }$ and $Z ^ { + } = \rho ^ { + } c ^ { + }$ are introduced and $v _ { n } ^ { + } = \mathbf { v } ^ { + } \cdot \mathbf { n } ^ { + } =$ $- \mathbf { v } ^ { + } \cdot \mathbf { n } , v _ { n } ^ { - } = \mathbf { v } ^ { - } \cdot \mathbf { n }$ . Substitution of the calculated ?? from Eq. (19) into Eq. (18) yields the Riemann solutions $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ }. For the element in the inward direction of ??, the upwind numerical flux ?? along ?? reads

$$
\mathbf {F} = \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} = \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {-} + c ^ {-} \alpha_ {1} \mathbf {r} _ {1} ^ {-},\tag{20}
$$

而对于界面另一侧的单元，迎风数值通量 ??沿着 ？？是

while, for the element on the other side of the interface, the upwind numerical flux ?? along ?? is

$$
\mathbf {F} = \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {+} + c ^ {+} \alpha_ {4} \mathbf {r} _ {4} ^ {+}.\tag{21}
$$

现在返回到声波穿过覆盖的多孔材料界面传播的物理设置。所有带有上标 $^ { 6 6 } - { } ^ { 5 5 }$ 和 $" + "$ 的量均使用空气和材料域的值进行评估，并用下标 ?? 表示和 $m ,$ 分别。对于沿空气域边界 $\varOmega _ { a } ,$ 的单元，$\boldsymbol { \mathbf { n } } _ { a } : = \boldsymbol { \mathbf { n } } = - \boldsymbol { \mathbf { n } } _ { m }$ 方向上的迎风数值通量 $\mathbf { F } ^ { a }$ 遵循 (20) 并可以紧凑地写为

Now return back to the physical setting of the acoustic wave propagation across the covered porous material interface. All quantities with superscripts $^ { 6 6 } - { } ^ { 5 5 }$ and $" + "$ are evaluated using the values of air and material domain, and denoted with subscripts ?? and $m ,$ respectively. For the elements along the air domain boundary $\varOmega _ { a } ,$ the upwind numerical flux $\mathbf { F } ^ { a }$ in the direction of $\boldsymbol { \mathbf { n } } _ { a } : = \boldsymbol { \mathbf { n } } = - \boldsymbol { \mathbf { n } } _ { m }$ follows (20) and can be written compactly as

$$
\mathbf {F} ^ {a} = \mathbf {R} _ {a} \boldsymbol {\Lambda} _ {a} \left[ \begin{array}{c} \mathcal {R} _ {a m} \varpi_ {a} ^ {o} + \mathcal {T} _ {m a} \varpi_ {m} ^ {o} \\ 0 \\ 0 \\ \varpi_ {a} ^ {o} \end{array} \right],\tag{22}
$$

在哪里

where

$$
\begin{array}{r} \varpi_ {a} ^ {o} = \frac {p _ {a}}{Z _ {a}} + \mathbf {v} _ {a} \cdot \mathbf {n} _ {a}, \\ \varpi_ {m} ^ {o} = \frac {p _ {m}}{Z _ {m}} + \mathbf {v} _ {m} \cdot \mathbf {n} _ {m}, \end{array}\tag{23a}
$$

(23b) 

分别是从空气侧和多孔材料侧向界面传播的出射特征波。这里有两个系数

are the outgoing characteristic waves traveling towards the interface from the air and the porous material side, respectively. Here, two coefficients

$$
\mathcal {R} _ {a m} = \frac {Z _ {t} + Z _ {m} - Z _ {a}}{Z _ {t} + Z _ {m} + Z _ {a}}, \quad \mathcal {T} _ {m a} = \frac {2 Z _ {m}}{Z _ {t} + Z _ {m} + Z _ {a}},\tag{24}
$$

引入并命名为从空气到材料域的反射系数$\mathcal { R } _ { a m }$和从材料到空气域的透射系数$\tau _ { m a }$。这些系数的物理解释将在第 3.2.1 节中进一步讨论。

are introduced and named as the reflection coefficients $\mathcal { R } _ { a m }$ from air to material domain and the transmission coefficients $\tau _ { m a }$ from material to air domain. The physical interpretations of these coefficients will be discussed further in Section 3.2.1.

类似地，对于材料域内的单元，由于法向量的符号变化，沿${ \mathbf { n } } _ { m }$的迎风数值通量$\mathbf { F } ^ { m }$为式(21)的负值，即

Similarly, for the element inside the material domain, the upwind numerical flux $\mathbf { F } ^ { m }$ along ${ \mathbf { n } } _ { m }$ is the negative of expression (21) due to the sign change of the normal vector, that is

$$
\mathbf {F} ^ {m} = \mathbf {R} _ {m} \boldsymbol {\Lambda} _ {m} \left[ \begin{array}{c} \mathcal {R} _ {m a} \varpi_ {m} ^ {o} + \mathcal {T} _ {a m} \varpi_ {a} ^ {o} \\ 0 \\ 0 \\ \varpi_ {m} ^ {o} \end{array} \right],\tag{25}
$$

在哪里

where

$$
\mathcal {R} _ {m a} = \frac {Z _ {t} + Z _ {a} - Z _ {m}}{Z _ {t} + Z _ {m} + Z _ {a}}, \quad \mathcal {T} _ {a m} = \frac {2 Z _ {a}}{Z _ {t} + Z _ {m} + Z _ {a}}.\tag{26}
$$

## 3.2.1. <sup></sup> 和 <sup></sup> 的物理解释与统一数值公式

## 3.2.1. Physical interpretations of <sup></sup> and <sup></sup> and unified numerical formulations

特征概念在数值求解线性双曲系统初边值问题中发挥着关键作用，因为它从根本上将系统的代数结构与所代表的波传播物理现象联系起来。从理论简正模态分析可知[66-68]，系统的特征变化允许以所有波数的平面波形式进行模态的解展开。因此，从物理角度来看，传入和传出的特征波分量可以解释为平面波[69]。出于同样的原因，方程中引入的系数<sup></sup>和<sup></sup>。 （24）和（26）与穿过阻抗不连续界面的平面波的粒子速度的相应平面波反射和传输系数完全匹配[70]。

The concept of characteristics plays a pivotal role in numerically solving initial–boundary value problems of linear hyperbolic system since it fundamentally connects the algebraic structure of the system with the represented physical phenomena of wave propagation. It is known from the theoretical normal mode analysis [66–68] that the characteristic variety of the system admits solution expansions of modes in the form of plane waves of all wave numbers. Consequently, the incoming and outgoing characteristic wave components can be interpreted as plane waves from a physical point of view [69]. For the same reason, the coefficients <sup></sup> and <sup></sup> introduced in Eqs. (24) and (26) match exactly the corresponding plane wave reflection and transmission coefficients of the particle velocity of plane waves traveling across an impedance discontinuity interface [70].

当波在均匀介质内传播时，不会发生反射。然后，<sup></sup> 减少到零，并且 <sup></sup> 保持一致。例如，对于空气子域内的单元 $\varOmega ^ { e }$，内部单元表面 $\partial \mathcal { Q } ^ { e }$ 处的上风通量 ${ \bf F } ^ { e } ( { \bf q } _ { h } ^ { e } , { \bf q } _ { h } ^ { e + } )$ 变为：

When the wave propagates inside a homogeneous medium, it experiences no reflection. Then, <sup></sup> reduces to zero and <sup></sup> remains unity. For example, for a element $\varOmega ^ { e }$ inside the air subdomain, the upwind flux ${ \bf F } ^ { e } ( { \bf q } _ { h } ^ { e } , { \bf q } _ { h } ^ { e + } )$ at the interior element surface $\partial \mathcal { Q } ^ { e }$ becomes:

$$
\mathbf {F} ^ {e} (\mathbf {q} _ {h} ^ {e}, \mathbf {q} _ {h} ^ {e +}) = \mathbf {R} _ {a} \boldsymbol {\Lambda} _ {a} \left[ \begin{array}{c} \frac {p _ {h} ^ {e +}}{Z _ {a}} - \mathbf {v} _ {h} ^ {e +} \cdot \mathbf {n} _ {e} \\ 0 \\ 0 \\ \frac {p _ {h} ^ {e}}{Z _ {a}} + \mathbf {v} _ {h} ^ {e} \cdot \mathbf {n} _ {e} \end{array} \right].\tag{27}
$$

因此，对于内部单元和边界单元都可以实现精确迎风通量的统一数值处理。

Therefore, a unified numerical treatment of exact upwind flux can be realized for both interior elements and boundary elements.

## 3.2.2. 纳入频率依赖性

## 3.2.2. Incorporation of frequency-dependency

到目前为止，所提出的公式仅限于具有与频率无关的实值传输阻抗 $Z _ { t } .$ 的覆盖界面。然而，覆盖材料通常表现出与频率相关的声学特性，并且 $Z _ { t } , e.g.$ 存在各种复杂程度的模型，这是微穿孔板的经典 Maa 模型 [71]。为了合并通用宽带模型，采用了与之前的工作[69]相同的数值策略。首先，类似于等式中对 $\rho _ { \mathrm { e f } } ( \omega )$ 和 $c _ { \mathrm { e f } } ( \omega )$ 的数值处理。 (4)、平面波系数$\mathcal { R } _ { m a } ( \omega ) , \mathcal { R } _ { a m } ( \omega ) , \mathcal { T } _ { m a } ( \omega )$和$\tau _ { a m } ( \omega )$在频域中以多极点模型的形式表示。然后，这些系数与相应的输出特征波在时间上进行卷积，如方程（1）中所定义。 (22)和(25)获得入射波，其中使用ADE方法计算卷积。

So far, the presented formulations are limited to the covering interface with a frequency-independent real-valued transfer impedance $Z _ { t } .$ . However, covering materials typically exhibit frequency-dependent acoustic properties and models of various levels of complexity exist for $Z _ { t } , e.g.$ ., the classical Maa’s model for microperforated panels [71]. In order to incorporate general broadband models, the same numerical strategy from previous work [69] is adopted. First, similar to the numerical treatments to $\rho _ { \mathrm { e f } } ( \omega )$ and $c _ { \mathrm { e f } } ( \omega )$ in Eqs. (4), the plane wave coefficients $\mathcal { R } _ { m a } ( \omega ) , \mathcal { R } _ { a m } ( \omega ) , \mathcal { T } _ { m a } ( \omega )$ and $\tau _ { a m } ( \omega )$ are expressed in the form of the multi-pole model in the frequency-domain. Then, these coefficients are convolved in time with the corresponding outgoing characteristic waves as defined in Eqs. (22) and (25) to obtain the incoming waves, during which the ADE method is used to calculate the convolutions.

在这项工作中，柔软的渗透膜模型被认为是一个代表性的例子。其声学特性主要由两个材料参数决定，即用 ?? 表示的表面质量密度。 [kg $\mathrm { m } ^ { - 2 } ]$ 和流阻 $r _ { f } \ [ \mathrm { P a } \ s \ \mathrm { m } ^ { - 1 } ]$ 分别代表声振和透气性的影响。类似于两个并联电路元件的电阻抗，传输阻抗为 [56]

In this work, a limp permeable membrane model is considered as a representative example. Its acoustic characteristics are mainly governed by two material parameters, namely the surface mass density denoted by ?? [kg $\mathrm { m } ^ { - 2 } ]$ and the flow resistance $r _ { f } \ [ \mathrm { P a } \ s \ \mathrm { m } ^ { - 1 } ]$ representing the effects of sound induced vibration and air permeability respectively. Analogous to the electric impedance of two circuit elements in parallel, the transfer impedance reads [56]

$$
Z _ {t} (\omega) = \left(\frac {1}{r _ {f}} + \frac {1}{\mathrm{i} \omega m}\right) ^ {- 1}.\tag{28}
$$

将 $Z _ { t } ( \omega )$ 代数运算成方程。 (24)和(26)屈服系数<sup></sup>和<sup></sup>以实极模型的形式表示为

Algebraic manipulations of $Z _ { t } ( \omega )$ into Eqs. (24) and (26) yield coefficients <sup></sup> and <sup></sup> in the form of real pole models as

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

与 $B _ { t } = 2 / ( r _ { f } + Z _ { a } + Z _ { m } )$ 和 $\zeta _ { t } = r _ { f } ( Z _ { a } + Z _ { m } ) / ( m Z _ { a } + m Z _ { m } + m r _ { f } )$

with $B _ { t } = 2 / ( r _ { f } + Z _ { a } + Z _ { m } )$ and $\zeta _ { t } = r _ { f } ( Z _ { a } + Z _ { m } ) / ( m Z _ { a } + m Z _ { m } + m r _ { f } )$

## 3.3. 半离散公式

## 3.3. Semi-discrete formulation

半离散公式是通过代入节点基础展开式得到的。 (12) 并将导出的数值通量公式代入方程的强公式。 （13）。让向量 ${ \bf v } _ { x h } ^ { e } , { \bf v } _ { y h } ^ { e } , { \bf v } _ { z h } ^ { e }$ 和 ${ \bf p } _ { h } ^ { e }$ 分别表示元素 $\varOmega ^ { e }$ 中的所有未知主节点值 $v _ { x h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) , v _ { y h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) , v _ { z h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t )$ 和 $p _ { h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t )$ ，例如 $\mathbf { v } _ { x h } ^ { e } = [ v _ { x h } ^ { e } ( \mathbf { x } _ { 1 } ^ { e } , t ) , v _ { x h } ^ { e } ( \mathbf { x } _ { 2 } ^ { e } , t ) , \dots , v _ { x h } ^ { e } ( \mathbf { x } _ { N _ { v } } ^ { e } , t ) ] ^ { T }$ 。如果元素$\varOmega ^ { e }$在空气域，则得到如下矩阵形式：

The semi-discrete formulation is obtained by substituting the nodal basis expansion Eq. (12) and the derived numerical flux formulation into the strong formulation of Eq. (13). Let vectors ${ \bf v } _ { x h } ^ { e } , { \bf v } _ { y h } ^ { e } , { \bf v } _ { z h } ^ { e }$ and ${ \bf p } _ { h } ^ { e }$ represent all the unknown primary nodal values $v _ { x h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) , v _ { y h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) , v _ { z h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t )$ and $p _ { h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t )$ in the element $\varOmega ^ { e }$ , respectively, e.g., $\mathbf { v } _ { x h } ^ { e } = [ v _ { x h } ^ { e } ( \mathbf { x } _ { 1 } ^ { e } , t ) , v _ { x h } ^ { e } ( \mathbf { x } _ { 2 } ^ { e } , t ) , \dots , v _ { x h } ^ { e } ( \mathbf { x } _ { N _ { v } } ^ { e } , t ) ] ^ { T }$ . If the element $\varOmega ^ { e }$ is in the air domain, the following matrix form is obtained:

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

第二个上标在哪里？ $\varOmega ^ { e }$ 表示单元的第 $\partial \mathcal { Q } ^ { e r }$ 面，$f$ 是单元 $\varOmega ^ { e }$ 的总面数，对于四面体单元来说等于 4 $\hat { \mathbf { F } } _ { v _ { x } } ^ { e r } , \hat { \mathbf { F } } _ { v _ { v } } ^ { e r } , \hat { \mathbf { F } } _ { v _ { z } } ^ { e r }$ 和 $\hat { \mathbf { F } } _ { p } ^ { e r }$ 是数值通量分量的节点值向量，由式（1）计算得出。 (27) 对于内部元件表面或方程。 (22) 为边界面。单元质量矩阵 $\mathbf { M } ^ { e } ,$ 、单元刚度矩阵 ${ \bf S } _ { j } ^ { e }$ 和单元面矩阵 ${ { \bf { M } } ^ { e r } }$ 定义为：

where the second superscript ?? denotes the ??th surface $\partial \mathcal { Q } ^ { e r }$ of the element $\varOmega ^ { e }$ and $f$ is the total number of faces of the element $\varOmega ^ { e }$ , which is equal to 4 for tetrahedra elements $\hat { \mathbf { F } } _ { v _ { x } } ^ { e r } , \hat { \mathbf { F } } _ { v _ { v } } ^ { e r } , \hat { \mathbf { F } } _ { v _ { z } } ^ { e r }$ and $\hat { \mathbf { F } } _ { p } ^ { e r }$ are the nodal value vectors of the numerical flux component, which are calculated by Eq. (27) for interior element surfaces or Eq. (22) for boundary surfaces. The element mass matrix $\mathbf { M } ^ { e } ,$ , the element stiffness matrices ${ \bf S } _ { j } ^ { e }$ and the element face matrices ${ { \bf { M } } ^ { e r } }$ are defined as:

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

其中 $j ~ ( j \in \{ x , y , z \} )$ 是笛卡尔坐标，?? $N _ { f p }$ 是沿一个单元面的节点数。如果元素 $\varOmega ^ { e }$ 在材料域中，则得到以下矩阵形式：

where $j ~ ( j \in \{ x , y , z \} )$ is the Cartesian coordinate and ?? $N _ { f p }$ is the number of nodes along one element face. If the element $\varOmega ^ { e }$ is in the material domain, the following matrix form is obtained:

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

其中向量 $\phi _ { \rho k } ^ { x e } , \phi _ { \rho k } ^ { y e } , \phi _ { \rho k } ^ { z e }$ 和 $\phi _ { C k } ^ { e }$ 表示辅助变量的局部未知节点值向量，这些辅助变量通过方程式与主节点值耦合。 （7）。

where vectors $\phi _ { \rho k } ^ { x e } , \phi _ { \rho k } ^ { y e } , \phi _ { \rho k } ^ { z e }$ , and $\phi _ { C k } ^ { e }$ denote the local unknown nodal value vectors of auxiliary variables that are coupled with primary nodal values by Eq. (7).

## 3.4. 时间离散化

## 3.4. Temporal discretization

通过DG方法进行空间离散后，全半离散系统可以用ODE的一般形式表示为：

After the spatial discretization by the DG method, the total semi-discrete system can be expressed in a general form of ODEs as:

$$
\frac {\partial \tilde {\mathbf {q}} _ {h}}{\partial t} = \mathcal {L} \big (\tilde {\mathbf {q}} _ {h} (t), t \big),\tag{33}
$$

其中 ${ \tilde { \mathbf { q } } } _ { h }$ 是未知向量，包括等式中的主要声学变量 ${ \bf q } _ { h }$ 和辅助变量 $\{ \phi _ { \rho k } , \phi _ { C k } \}$。 (7).这里，<sup></sup>是考虑方程2的DG空间离散化的算子。 (13) 和辅助微分方程。 （7）。该系统使用泰勒级数时间积分器（TSI）方案进行时间积分[72]，得到最终的时间离散公式：

where ${ \tilde { \mathbf { q } } } _ { h }$ is the unknown vector including primary acoustic variables ${ \bf q } _ { h }$ and auxiliary variables $\{ \phi _ { \rho k } , \phi _ { C k } \}$ from Eqs. (7) . Here, <sup></sup> is the operator that considers both the DG spatial discretization of Eq. (13) and auxiliary differential Eqs. (7). This system is temporally integrated using the Taylor Series time integrator (TSI) scheme [72], resulting in a final time-discrete formulation as

$$
\tilde {\mathbf {q}} _ {h} (t + \Delta t) = \tilde {\mathbf {q}} _ {h} (t) + \sum_ {i = 1} ^ {N _ {t}} \frac {\Delta t ^ {i}}{i !} \mathcal {L} ^ {i} \tilde {\mathbf {q}} _ {h} (t),\tag{34}
$$

其中 $N _ { t }$ 表示准确性的时间顺序，$\varDelta t = t ^ { n + 1 } - t ^ { n }$ 是时间步长。

where $N _ { t }$ denotes the temporal order of accuracy and $\varDelta t = t ^ { n + 1 } - t ^ { n }$ is the time step.

作为一种显式时间步长方法，TSI 方案受 Courant-Friedrichs-Lewy (CFL) 条件稳定性的影响，该稳定性设置了时间步长的上限。对于采用 DG 方法的空间 1 阶离散化组合和采用 TSI 方案的时间 (??+1) 阶离散化组合，时间步长大小如下 [73]

As an explicit time-stepping method, the TSI scheme is subject to the Courant–Friedrichs–Lewy (CFL) conditional stability that sets an upper bound on the time step size. For a discretization combination of ??th order in space with the DG method and a (?? +1)th order in time with the TSI scheme, the time step size follows [73]

$$
\Delta t = C _ {C F L} \Delta x _ {l} \frac {1}{c} \frac {1}{(2 N + 1)},\tag{35}
$$

其中 $C _ { C F L }$ 是 <sup></sup>(1) 阶常量，??是介质的恒定波速，$\varDelta x _ { l }$ 是单元尺寸的度量。对于多孔材料域，除了 DG 空间算子之外，时间步长还受到辅助微分方程 $i e .$ 的刚度、拟合参数 $\{ \zeta _ { \rho } , \zeta _ { C } \}$ 的限制，如方程 1 所示。 （4）。研究发现[74]，当使用矢量拟合时，$\zeta _ { \rho }$和$\zeta _ { C }$的最大值随着极数的增加而增加。为了提高计算效率，采用了伴随 TSI 方案[72]的显式局部时间步进策略。因此，可以在多孔材料子域中使用较小的时间步长，以确保乘积 ?? ⋅ ????落入时间积分方案的稳定区域。 TSI 方案与多级 Runge-Kutta 方案 [75] 类似，随着精度 $N _ { t }$ 数量级的增加，具有更大的稳定区域。例如，四阶 TSI 方案允许最大值为 $\zeta \cdot \varDelta t = 2 . 7 7 5$ 。关于数值稳定性和实现细节的进一步讨论可以在参考文献中找到。 [72]。

where $C _ { C F L }$ is a constant of order <sup></sup>(1), ?? is the constant wave speed of the medium, and $\varDelta x _ { l }$ is a measure of the element size. For the porous material domain, apart from the DG spatial operator, the time step size is additionally bounded by the stiffness of auxiliary differential equations, $i e .$ , the fitting parameters $\{ \zeta _ { \rho } , \zeta _ { C } \}$ as in Eqs. (4). It was found [74] that when vector fitting is used, the maximum value of $\zeta _ { \rho }$ and $\zeta _ { C }$ increases with the number of poles. To improve the computational efficiency, an explicit local time-stepping strategy accompanying the TSI scheme [72] is employed. Consequently, a smaller time step size can be used in the porous material subdomain to ensure that the product ?? ⋅ ???? falls into the stability region of the time-integration scheme. The TSI scheme, similar to the multi-stage Runge–Kutta scheme [75], has a larger stability region with increasing order of accuracy $N _ { t }$ . For example, a fourth-order TSI scheme admits a maximum value of $\zeta \cdot \varDelta t = 2 . 7 7 5$ . Further discussion on numerical stability and details of implementations can be found in the Ref. [72].

## 4. 数值结果

## 4. Numerical results

## 4.1. 多孔材料的极点识别

## 4.1. Poles identification of porous materials

在这项工作中，考虑了两种由 JCAL 模型表征的刚性框架多孔材料：三聚氰胺泡沫和玻璃棉。这些材料用于早期研究[7,53]。表 1 显示了它们的模型参数。作为参考，它们的理论正入射吸声系数如图 2 所示。它们的有效密度 $\rho _ { \mathrm { e f } }$ 和有效压缩率 $c _ { \mathrm { e f } }$ 作为角频率 ?? 的函数。被描述为[29]

In this work, two kinds of rigid-frame porous material characterized by JCAL model are considered: melamine foam and glass wool. These materials are used in earlier studies [7,53]. Table 1 shows their model parameters. As a reference, their theoretical normal incidence sound absorption coefficient are shown in Fig. 2. Their effective density $\rho _ { \mathrm { e f } }$ and effective compressibility $c _ { \mathrm { e f } }$ as a function of angular frequency ?? are described as [29]

$$
\begin{array}{r l} & {\rho_ {\mathrm{ef}} (\omega) = \frac {\rho_ {a} \alpha_ {\infty}}{\varphi} \Big [ 1 + \frac {\sigma \varphi}{\mathrm{i} \omega \alpha_ {\infty} \rho_ {a}} \big (1 + \frac {4 \mathrm{i} \alpha_ {\infty} ^ {2} \eta \rho_ {a}}{\sigma^ {2} \Lambda^ {2} \varphi^ {2}} \big) ^ {1 / 2} \Big ],} \\ & {\mathcal {C} _ {\mathrm{ef}} (\omega) = \frac {\varphi}{\rho_ {a} c _ {a} ^ {2}} \Bigg (\gamma - \frac {\gamma - 1}{\big [ 1 + \frac {\varphi \eta}{\mathrm{i} \omega k _ {0} ^ {\prime} \rho_ {a} P _ {r}} (1 + \frac {4 \mathrm{i} \omega k _ {0} ^ {\prime 2} \rho_ {a} P _ {r}}{\eta \Lambda^ {\prime 2} \varphi^ {2}}) ^ {1 / 2} \big ]} \Bigg).} \end{array}\tag{36a}
$$

(36b) 

方程中的极限常数值。 (4) 可以解析计算为 $\rho _ { m } = \rho _ { a } \alpha _ { \infty } / \varphi$ 和 $C _ { m } = \varphi / ( \rho _ { a } c _ { a } ^ { 2 } )$ 。 VF算法输入所需的频率矢量设置为[20∶2000]Hz，分辨率为1Hz。正如对耗散多孔材料所预期的那样，VF 算法仅自动选择真实极点。与参考文献的发现相同。 [53]，数值测试表明相对根平方误差随着极数 $\{ N _ { \rho } , N _ { C } \}$ 的增加而快速减小，并且当 $\{ N _ { \rho } , N _ { C } \} > 5$ 时下降到 $1 0 ^ { - 5 }$ 以下

The limiting constant values in Eqs. (4) can be calculated analytically as $\rho _ { m } = \rho _ { a } \alpha _ { \infty } / \varphi$ and $C _ { m } = \varphi / ( \rho _ { a } c _ { a } ^ { 2 } )$ . The frequency vector needed as input for the VF algorithm is set to [20 ∶ 2000] Hz with a resolution of 1 Hz. As expected for dissipative porous materials, the VF algorithm selects only real poles automatically. Same as findings from Ref. [53], numerical tests show that the relative root squared error diminishes fast with increasing number of poles $\{ N _ { \rho } , N _ { C } \}$ , and drops below $1 0 ^ { - 5 }$ when $\{ N _ { \rho } , N _ { C } \} > 5$

表1

Table 1



多孔材料的特性

Properties of porous materials

<table><tr><td>性能</td><td>玻璃棉</td><td>三聚氰胺泡沫</td></tr><tr><td>流动电阻率σ [N s m-4]</td><td>70821</td><td>4500</td></tr><tr><td>孔隙率φ</td><td>0.967</td><td>0.99</td></tr><tr><td>曲折度α∞</td><td>1.049</td><td>1.0</td></tr><tr><td>粘性长度Λ [m]</td><td>6 × 10-5</td><td>1.3×10-4</td></tr><tr><td>热长度Λ'[m]</td><td>1.4×10-4</td><td>1.6×10-4</td></tr><tr><td>热导率k0' [m2]</td><td>6.345×10-9</td><td>4.0×10-9</td></tr><tr><td>动力粘度η[N·m-2]</td><td>1.82×10-5</td><td>1.82× 10-5</td></tr><tr><td>普朗特数Pr</td><td>0.71</td><td>0.71</td></tr></table>

<table><tr><td>Property</td><td>Glass wool</td><td>Melamine foam</td></tr><tr><td>Flow resistivity σ [N s m-4]</td><td>70821</td><td>4500</td></tr><tr><td>Porosity φ</td><td>0.967</td><td>0.99</td></tr><tr><td>Tortuosity α∞</td><td>1.049</td><td>1.0</td></tr><tr><td>Viscous length Λ [m]</td><td>6 × 10-5</td><td>1.3 × 10-4</td></tr><tr><td>Thermal length Λ&#x27; [m]</td><td>1.4 × 10-4</td><td>1.6 × 10-4</td></tr><tr><td>Thermal permeability k0&#x27; [m2]</td><td>6.345 × 10-9</td><td>4.0 × 10-9</td></tr><tr><td>Dynamic viscosity η [N m-2]</td><td>1.82 × 10-5</td><td>1.82 × 10-5</td></tr><tr><td>Prandtl number Pr</td><td>0.71</td><td>0.71</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/77b6caf1c45e9c315e7636c70a254a6b5c3d64f6b4938ef31550a568b099214f.jpg)



图2.模拟多孔材料的理论法向入射吸收系数。

Fig. 2. Theoretical normal incidence absorption coefficient of simulated porous materials.

## 4.2. 一维测试：针对解析解进行验证

## 4.2. 1D tests: verification against analytical solutions

为了验证所提出的数值方案，进行了一维数值测试。计算域包括空气子域 $x \in [ - 1 . 5 , 0 ]$ m 和由三聚氰胺泡沫制成的多孔材料 $x \in [ 0 , 1 . 5 ]$ m 子域，它们由 $x = 0 .$ 处的压力跳跃界面分隔开。在域的两端应用完全吸声边界条件。不失一般性，该界面的声学特征为第 3.2.2 节中介绍的柔软渗透膜，并具有表面质量密度 $m = 0 . 1 3$ kg $\mathrm { m } ^ { - 2 }$ 和流动阻力 $r _ { f } = 6 4 ~ \mathrm { N } ~ s ~ \mathrm { m } ^ { - 3 }$ ，这是通用织物的代表。这个数值测试的灵感来自于[39]的工作，其中详细提供了时间的解析解。空气子域中的模拟由墨西哥帽子脉冲启动：

To validate the proposed numerical scheme, 1D numerical tests are performed. The computational domain includes an air subdomain $x \in [ - 1 . 5 , 0 ]$ m and a porous material $x \in [ 0 , 1 . 5 ]$ m subdomain made of melamine foam, which are separated by a pressure jump interface at $x = 0 .$ . Totally sound absorbing boundary conditions are applied at both ends of the domain. Without loss of generality, the interface is acoustically characterized as the limp permeable membrane introduced in Section 3.2.2 and has a surface mass densit $m = 0 . 1 3$ kg $\mathrm { m } ^ { - 2 }$ and a flow resistance $r _ { f } = 6 4 ~ \mathrm { N } ~ s ~ \mathrm { m } ^ { - 3 }$ , which is representative of general purpose fabrics. This numerical test is inspired by the work of [39], where an analytical solution in time is provided in detail. The simulations in the air subdomain are initiated by a Mexican hat pulse:

$$
\begin{array}{r l} & p (x, t = 0) = \big (1 - \big [ \frac {x - x _ {s}}{B} \big ] ^ {2} \big) \mathrm{e} ^ {- (\frac {x - x _ {s}}{\sqrt {2} B}) ^ {2}}, \\ & u (x, t = 0) = \frac {1}{\rho_ {a} c _ {a}} \big (1 - \big [ \frac {x - x _ {s}}{B} \big ] ^ {2} \big) \mathrm{e} ^ {- (\frac {x - x _ {s}}{\sqrt {2} B}) ^ {2}}, \end{array}\tag{37a}
$$

(37b) 

它以 $x _ { s } = - 1$ m 为中心，具有高达 4000 Hz 的足够能量，宽度为 $B = 0 . 0 4 5$ m。在材料子域中，声学变量和辅助变量都初始化为零。

which is centered around $x _ { s } = - 1$ m and has sufficient energy up to 4000 Hz with width $B = 0 . 0 4 5$ m. In the material subdomain, both acoustic and auxiliary variables are initialized to zero.

单元尺寸设置为 $\varDelta x = 0 . 1$ m，空间和时间均采用 5 阶离散化方案，即三聚氰胺泡沫的 $N = N _ { t } = 5 .$ $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } }$ 用 6 个实极点近似，导致高达 4 kHz 幅度为 $\mathcal { O } ( 1 0 ^ { - 5 } )$ 的平方根相对误差。图3(a)描绘了三个时刻的压力场。 $\bar { t } = 1 . 5$ ms 时刻的脉冲是向界面传播的入射波。然后，在 $\bar { t } = 2 . 9$ ms 处，脉冲与界面相互作用，数值方案可以很好地捕获界面上的压力跃变。可以在 $\bar { t } = 3 . 6$ ms 处观察到反射脉冲和发射脉冲。图3（b）显示了相对于解析解的一致数量级的绝对误差。

The element size is set as $\varDelta x = 0 . 1$ m and 5th order discretization schemes are used for both space and time, i.e., $N = N _ { t } = 5 .$ $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ of the melamine foam are approximated with 6 real poles, resulting in a root squared relative error of magnitude $\mathcal { O } ( 1 0 ^ { - 5 } )$ up to 4 kHz. Fig. 3(a) depicts the pressure fields at three time instants. The pulse at the time $\bar { t } = 1 . 5$ ms is the incident wave traveling towards the interface. Then, at $\bar { t } = 2 . 9$ ms, the pulse interacts with the interface, across which a pressure jump is well captured by the numerical scheme. The reflected pulse and the transmitted pulse can be observed at $\bar { t } = 3 . 6$ ms. Fig. 3(b) shows the absolute error of a consistent order of magnitude with respect to the analytical solutions.

为了进行严格的误差分析，数值解的误差被分成来自两个来源的两部分，即由等式中的$\rho _ { \mathrm { e f } }$和$c _ { \mathrm { e f } }$多极模型近似截断引起的模型误差。 (4)、以及空间和时间离散化的数值误差。具体来说，假设 $p _ { \mathrm { n u m } }$ 表示数值解，${ p } _ { \mathrm { a n a } }$ 是根据 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } } ,$ 的精确值计算的解析解，$P _ { \mathrm { a n a * } }$ 是根据 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } }$ 的截断多极模型表达式构建的解析解。这两种类型的一次相对误差??<sup>̄</sup>由[39]全局测量：

Aiming for a rigorous error analysis, the error of the numerical solution is split into two parts from two sources, namely the model error induced by the truncation of multipole model approximation of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ from Eq. (4), and the numerical error from the spatial and temporal discretization. To be specific, suppose $p _ { \mathrm { n u m } }$ denotes the numerical solution, ${ p } _ { \mathrm { a n a } }$ is the analytical solution calculated based on the exact values of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } } ,$ and $P _ { \mathrm { a n a * } }$ is the analytical solution built from truncated multipole model expressions of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ . These two types of relative error at a time ??<sup>̄</sup> are measured globally by [39]:

$$
\epsilon_ {\mathrm{num}} (\bar {t}) = \frac {\| p _ {\mathrm{ana*}} (\bar {t}) - p _ {\mathrm{num}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana*}} (\bar {t}) \| _ {L ^ {2}}}\tag{38}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/efe34afcf5532b448fc874208cafc29467f9a5c54bfcc554652d548cd6542686.jpg)



（一个）

(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/a4a5f9568f306a880462f2d6b609d1d7673934192eac686a2d0682308a7bd1e7.jpg)



(二)

(b)



图 3 (a) 空气和覆盖有织物的三聚氰胺泡沫在三个时刻的模拟压力场； (b) 压力场误差。

Fig. 3. (a) Simulated pressure field in the air and the melamine foam with fabrics covering at three time instants; (b) error of the pressure field.


$$
\epsilon_ {\mathrm{model}} (\bar {t}) = \frac {\| p _ {\mathrm{ana*}} (\bar {t}) - p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}},\tag{39}
$$

其中 $\begin{array} { r } { \| f \| _ { L ^ { 2 } } = ( \int _ { \Omega } f ^ { 2 } ( \mathbf x ) \mathrm { d } \mathbf x ) ^ { 1 / 2 } } \end{array}$ 表示 $L ^ { 2 }$ 范数，并使用多项式基础计算至近似阶。总相对误差通过以下方式全局测量

where $\begin{array} { r } { \| f \| _ { L ^ { 2 } } = ( \int _ { \Omega } f ^ { 2 } ( \mathbf x ) \mathrm { d } \mathbf x ) ^ { 1 / 2 } } \end{array}$ denotes the $L ^ { 2 }$ norm and is calculated with the polynomial basis up to the order of approximation. The total relative error is measured globally by

$$
\epsilon_ {\mathrm{tot}} (\bar {t}) = \frac {\| p _ {\mathrm{ana}} (\bar {t}) - p _ {\mathrm{num}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}}\tag{40}
$$

为了验证全局收敛速度，使用各种尺寸的均匀网格对域进行离散化。空气子域中的时间步长 $\varDelta t _ { a }$ 与网格大小成比例变化？？？遵循 CFL 条件 (35)，并将 $C _ { C F L }$ 保持为常数 0.5。为了适应极参数$\{ \zeta _ { \rho } , \zeta _ { C } \}$的高刚度，使用局部时间步进方案[72]，并将材料子域中的时间步长设置为$\Delta t _ { m } = 1 / 4 \Delta t _ { a }$以确保时间积分稳定。作为演示示例，图 4 显示了空间和时间均使用 5 阶离散化方案（即 $N = N _ { t } = 5$）的收敛图，其中描绘了（a）空气中的反射波和（b）多孔材料中的透射波在 $\bar { t } = 4 . 4$ ms 时刻的全局误差。对于较大的网格尺寸和时间步长，总误差主要由数值误差决定，而对于较小的网格尺寸，总误差收敛于建模误差。正如预期的那样，数值误差与极数无关。如图 4 所示，数值误差的收敛速度与 DG 方法所谓的“最佳”收敛速度 $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ 相匹配。

To verify the global convergence rate, the domain is discretized with uniform mesh of various sizes. The time step $\varDelta t _ { a }$ in air subdomain varies proportionally to the mesh size ???? following the CFL condition (35) with $C _ { C F L }$ kept as a constant of 0.5. To accommodate the high stiffness of pole parameters $\{ \zeta _ { \rho } , \zeta _ { C } \}$ , the local time-stepping scheme [72] is used and the time step size in the material subdomain is set as $\Delta t _ { m } = 1 / 4 \Delta t _ { a }$ to ensure that the time integration is stable. As an example of demonstration, the convergence plot of using 5th order discretization schemes for both space and time, i.e., $N = N _ { t } = 5$ are shown in Fig. 4, where the global error at the time $\bar { t } = 4 . 4$ ms for both (a) the reflected wave in the air, and (b) the transmitted wave in the porous material are depicted. For larger mesh sizes and time steps, the total error is dominated by the numerical error, whereas for smaller mesh sizes, the total error converges to the modeling error. As expected, the numerical error is independent of the number of poles. As shown in Fig. 4, the convergence rate of numerical error matches the so-called ‘‘optimal’’ rate of convergence $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ for the DG method.

为了进一步评估该方案，压力跃变界面周围的局部数值精度通过以下误差测量进行量化：

For further assessments of the scheme, the local numerical accuracy around the pressure jump interface is quantified with the following error measure:

$$
\epsilon_ {\mathrm{num}} ^ {l} (\bar {t}) = \frac {\left(\int_ {0} ^ {\bar {t}} | \lceil p \rceil_ {\mathrm{num}} - \lceil p \rceil_ {\mathrm{ana*}} | ^ {2} \mathrm{d} t\right) ^ {1 / 2}}{\left(\int_ {0} ^ {\bar {t}} \lceil p \rceil_ {\mathrm{ana*}} ^ {2} \mathrm{d} t\right) ^ {1 / 2}},\tag{41}
$$

其中$\lceil p \rceil ( t )$表示跨膜界面压力跃变的时间信号，并根据梯形法则计算时间积分。执行精度越来越高的仿真，极数固定为 5。图 5 描绘了 $\bar { t } = 4 . 4$ 毫秒时的局部数值误差 $\epsilon _ { \mathrm { { n u m } } } ^ { l } ( \bar { t } )$。正如我们所看到的，在精度的渐近范围内，实现了最佳速率 $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ )，而较粗网格和相应较大时间步长的误差逐渐遵循梯形求积规则的收敛速率。因此，所提出的接口数值处理保持了方案的精度顺序。用其他材料性能值进行数值试验也可以得出相同的结论。

where $\lceil p \rceil ( t )$ represents the time signal of the pressure jump across the interface of membrane, and the integral in time is calculated with the trapezoid rule. Simulations with an increasing number of order of accuracy are performed and the number of poles is fixed at 5. Fig. 5 depicts local numerical error $\epsilon _ { \mathrm { { n u m } } } ^ { l } ( \bar { t } )$ at $\bar { t } = 4 . 4$ ms. As we can see, in the asymptotic range of accuracy, the optimal rate $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ ) is achieved, whereas the error from coarser meshes and correspondingly larger time steps gradually follow the convergence rate of the trapezoid quadrature rule. Therefore, the proposed numerical treatment of the interface maintains the order of accuracy of the scheme. The same conclusions can be drawn from numerical tests with other material property values.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/2aac02e1de25a618b97144787e390ead1e0c5424968ad2b9553244a90bf4af1c.jpg)



（一个）

(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/dd042b9477c9bbefdbeb06c111e65102809a41abf685c076dfc1b4fea2f13d32.jpg)



(二)

(b)



图 4. 半无限空气和多孔材料的 $\epsilon _ { \mathrm { { t o t } } } , \epsilon _ { \mathrm { { n u m } } } ,$ 和 $\epsilon _ { m o d e l }$，其中 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } } .$ 的实极数不断增加。柔软的渗透界面产生（a）空气中的反射波和（b）多孔材料中的透射波。

Fig. 4. $\epsilon _ { \mathrm { { t o t } } } , \epsilon _ { \mathrm { { n u m } } } ,$ and $\epsilon _ { m o d e l }$ for semi-infinite air and porous material with increasing number of real poles of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } } .$ . The limp permeable interface begets (a) a reflected wave in the air, and (b) a transmitted wave in the porous material.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/5a902df616ae1c1e7d0a57acc262ecfeb134ca8667b275a4469f18820b7b78bf.jpg)



压力跳跃界面周围局部误差的收敛行为。

Convergence behavior of local error around pressure jump interface.

## 4.3. 刚性背衬多孔吸声体上方的球面波反射

## 4.3. Spherical wave reflection above rigidly-backed porous absorber

现在考虑单个球面波反射测试案例，其中单极源放置在多孔吸收体的刚性背衬层上方。这个基本案例旨在评估 3D 场景中所提出方案的准确性，并避免其他建模误差源，$_ { e.g. }$，几何误差。此外，Allard 等人求解了解析声压场。 [76]适用于多孔材料和空气之间存在自由界面的情况，因此被视为精确的参考解。如图 6 所示，自由界面跨越水平 ??–??飞机位于 $z = 0$ 。源放置在 $\mathbf { x _ { s } } = [ 0 , 0 , 0 . 3 5 ]$ m 处。两个接收器放置在 $\mathbf { x } _ { r 1 } = [ 0 , 0 , 0 . 0 2 ]$ m 和 $\mathbf { x } _ { r 2 } = [ 1 . 5 , 1 . 5 , 0 . 0 2 ]$ m 处，分别对应于正入射和斜入射（镜面反射角为 $8 0 ^ { \circ }$）的情况。高斯压力脉冲用于启动模拟

Now consider a single spherical wave reflection test case, where a monopole source is placed above a rigidly backed layer of porous absorber. This basic and fundamental case aims to assess the accuracy of the proposed scheme in 3D scenarios with other sources of modeling error avoided, $_ { e.g. }$ , geometrical error. Furthermore, an analytical sound pressure field solved by Allard et al. [76] is available for the case of a free interface between the porous material and the air, and is thus taken as an exact reference solution. As illustrated in Fig. 6, the free interface spans the horizontal ??–?? plane at $z = 0$ . A source is placed at $\mathbf { x _ { s } } = [ 0 , 0 , 0 . 3 5 ]$ m. Two receivers are placed at $\mathbf { x } _ { r 1 } = [ 0 , 0 , 0 . 0 2 ]$ m and $\mathbf { x } _ { r 2 } = [ 1 . 5 , 1 . 5 , 0 . 0 2 ]$ m, which corresponds to cases of a normal incidence and an oblique incidence with a specular reflection angle of $8 0 ^ { \circ }$ respectively. A Gaussian pressure pulse is used to initiate the simulations

$$
p (\mathbf {x}, t = 0) = \mathrm{e} ^ {\frac {- \ln 2}{b ^ {2}} (\mathbf {x} - \mathbf {x} _ {s}) ^ {2}},\tag{42a}
$$

$$
\mathbf {v} (\mathbf {x}, t = 0) = \mathbf {0},\tag{42b}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/2923e223c00c7d75c7238a955e8719375cbce39086f387ecef6d7c51d2009a8f.jpg)



图 6. 3D 测试的数值设置图示

Fig. 6. Illustration of the numerical setup for the 3D test


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/28ba7192d4c72db2cc80e564a357f081b8a77453b2e116fe7cbb5fb6c4e7b234.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/5d6dd3e1e6d727ae0207482e788b0ae07bf0a37e15d106761dac1e07b8eb69de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0d0dddcf450d1e6fc02f434169477afcde0eede08e85597df95dbbcf5f46071c.jpg)



（一个）

(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/be5a701817745e9df320538fb93bd21f3329d55da075c7cdf27ddbdf3412de75.jpg)



(二)

(b)



图 7. 三聚氰胺泡沫上方的模拟和分析压力场：(a) $\mathbf { x } _ { r 1 } , \theta ^ { \circ } = 0 ^ { \circ }$，(b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$

Fig. 7. Simulated and analytical pressures field above melamine foam at: (a) $\mathbf { x } _ { r 1 } , \theta ^ { \circ } = 0 ^ { \circ }$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$

半带宽值为 $b = 0 . 1 \textrm { m }$ ，表示源频谱高达 2 kHz。整个计算域的维度为 $[ - 3 , 4 ] \times [ - 3 , 4 ] \times [ - d , 3 . 5 ]$ m，其中 ??是多孔材料的厚度。使用网格划分软件 <sup>GMSH</sup> [77] 为空气和多孔材料生成符合四面体网格的非结构化几何形状。这里，网格尺寸受到多孔材料厚度的限制，该厚度等于四面体单元一侧的长度。为了获得足够的空间分辨率，使用七阶多项式基函数 $( N = 7 )$ 进行空间离散化，从而在 2 kHz 下每个波长产生大约 9 个体积平均自由度。时间顺序设置为 $N _ { t } = 5$ 以在模拟精度和效率之间取得良好的平衡。内切球的最小半径 min $( r _ { i n } )$ 用作单元尺寸 $\varDelta x _ { l }$ 的度量。在进行的数值试验中，单元尺寸是时间步长的主要限制因素。硬壁边界条件施加于整个计算域的外部边界，并且在来自外部边界的寄生反射到达之前接收器位置记录的压力信号被切断。

with a half-bandwidth value of $b = 0 . 1 \textrm { m }$ , indicating a source spectrum up to 2 kHz. The overall computational domain has a dimension of $[ - 3 , 4 ] \times [ - 3 , 4 ] \times [ - d , 3 . 5 ]$ m, where ?? is the thickness of the porous material. Unstructured geometry conforming tetrahedra meshes are generated with a meshing software <sup>GMSH</sup> [77] for both the air and the porous material. Here, the mesh sizes are constrained by the thickness of the porous material, which is equal to the length of one side of the tetrahedron elements. In order to have sufficient spatial resolution, 7th order polynomial basis functions $( N = 7 )$ are used for the spatial discretization, resulting in roughly 9 volume-averaged degrees of freedom per wavelength at 2 kHz. The temporal order is set at $N _ { t } = 5$ to achieve a decent balance between the simulation accuracy and efficiency. The minimum radius of the inscribed sphere min $( r _ { i n } )$ is used as a measure of the element size $\varDelta x _ { l }$ . Among the performed numerical tests, the element size is the dominant restriction factor on the time step size. Hard wall boundary conditions are imposed on exterior boundaries of the whole computational domain, and the recorded pressure signals at receivers’ locations are cut before spurious reflections from exterior boundaries arrive.

图 7 显示了模拟和分析压力谱 $\hat { p }$ 在振幅方面的比较 | ̂??|对于具有两种不同厚度的三聚氰胺泡沫的情况，相$\vartheta ( \hat { p } )$。解析解是通过使用 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } }$ 的精确值作为输入来计算的，而数值解是基于具有 5 个实极点的 $\rho _ { \mathrm { e f } }$ 和 $c _ { \mathrm { e f } }$ 的近似值。使用初始高斯脉冲的解析自由场时间解来标准化其非平坦源功率谱。模拟结果的大小已标准化，使得自由场解的形式为 $\mathrm { e } ^ { - \mathrm { i } k r } / ( 4 \pi r )$ 。同样，图8显示了当多孔吸收体由表1所示的玻璃棉表示时的比较结果。从这些结果可以看出，模拟解和参考解之间达到了很好的一致性，证明了所提出的边界方案在3D空间中的适用性以及在宽频率范围内的高精度。

Fig. 7 shows the comparison of the simulated and analytical pressure spectra $\hat { p }$ in terms of the amplitude | ̂??| and the phase $\vartheta ( \hat { p } )$ for the case of melamine foam with two different thicknesses. The analytical solutions are calculated by using the exact values of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ as inputs whereas the numerical solutions are built on approximations of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ with 5 real poles. The analytical free-field time solution of the initial Gaussian pulse is used in order to normalize its non-flat source power spectrum. The magnitude of simulated results have been normalized such that the free-field solution is of the form $\mathrm { e } ^ { - \mathrm { i } k r } / ( 4 \pi r )$ . Similarly, Fig. 8 displays the comparison results when the porous absorber is represented by the glass wool as shown in Table 1. It can be seen from these results that a good agreement between simulated and reference solutions is achieved, demonstrating the applicability of the proposed boundary scheme in 3D space and its high precision in a wide frequency range.

室内声学的特点是在外壳内发生多次反射。每次反射后都会累积额外的数值误差。为了在计算效率和准确性之间取得适当的平衡，重要的是根据自由度来量化误差，这与计算成本成正比。对于某个宽带入射声波，声压级损失和相位失真可以通过以下耗散误差测量和$\epsilon _ { a m p }$ [dB]相位误差测量$\epsilon _ { \vartheta }$ [%]来评估：

Room acoustics features multiple reflections happen inside an enclosure. Additional numerical error is accumulated after each reflection. In the interest of a decent balance between the computational efficiency and accuracy, it is important to quantify the error in terms of the degrees of the freedom, which is proportional to the computational cost. For a certain broadband incident acoustic wave, the loss of sound pressure level and the distortion of the phase can be evaluated by the following dissipation error measure and $\epsilon _ { a m p }$ [dB] phase error measure $\epsilon _ { \vartheta }$ [%]:

$$
\epsilon_ {\mathrm{amp}} (f) = 2 0 \log_ {1 0} \left| \frac {\hat {p} _ {\mathrm{num}} (f)}{\hat {p} _ {\mathrm{ana}} (f)} \right|,\tag{43a}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/f119484f68d86af3d00fd29c135eb83ce81257ac3c7ebdd92f1f865920259200.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/69e88d687eee10175ce7416ad49e56ee9980d9239eceec444d4df515f1f783d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0680e79aba9d688dca2ae324c8dc6f010da8a23ae4d942d6dc8a59c9c660915c.jpg)



（一个）

(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/bb626ceace6fe15218e1d69747cbf0f3395cb08b748c23a32c87735acf0096b1.jpg)



(二)

(b)



图 8. 玻璃棉上方的模拟和分析压力场：(a) $\mathbf { x } _ { r 1 } , \theta ^ { \circ } = 0 ^ { \circ }$，(b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$

Fig. 8. Simulated and analytical pressures field above glass wool at: (a) $\mathbf { x } _ { r 1 } , \theta ^ { \circ } = 0 ^ { \circ }$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/d2e9021351f062a4bfa038b9265e0359ae21ceaed40603481deacb29fb424f79.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/66ef5c2060a1ef0236d3d86b8632acb2cb9d6b98b9a6226b6b862ec7164b08d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0bf916740db2f59de5341a1620dbae12f919fe0d4ba019d3b98a04e0e242ee6d.jpg)



（一个）

(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/fd94b83c6ce20012c6178cea7b6ecca1f5da4a54f48bc714ad195f6680eec654.jpg)



(二)

(b)



图 9. 模拟的耗散误差 $\epsilon _ { a m p }$ [dB] 和相位误差 $\epsilon _ { \vartheta }$ [%]：(a) ?? $\theta ^ { \circ } = 0 ^ { \circ } ,$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$

Fig. 9. Dissipation error $\epsilon _ { a m p }$ [dB] and phase error $\epsilon _ { \vartheta }$ [%] of simulations at: (a) ?? $\theta ^ { \circ } = 0 ^ { \circ } ,$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$


$$
\epsilon_ {\vartheta} (f) = \frac {1}{\pi} \big (\vartheta \big (\hat {p} _ {\mathrm{num}} (f) \big) - \vartheta \big (\hat {p} _ {\mathrm{ana}} (f) \big) \big) \times 100 \%.\tag{43b}
$$

图 9 显示耗散误差 ??和相位误差？？两种多孔吸收体的模拟结果，其具有明显不同的 $\epsilon _ { a m p }$ $\epsilon _ { \vartheta }$ 不同的流阻值 $\sigma .$ 。网格和多项式阶次的组合导致每个波长在 2.kHz 下具有 9 个自由度的空间分辨率。正如预期的那样。当空间分辨率足够时，数值误差保持在可接受的水平。值得注意的是，低频范围（500 Hz 以下）的误差大于中频范围（500 – 1500 Hz）。这是由于记录的压力信号过早被切断，导致反射声场的低频功率损失。由于斜入射情况的第二个接收器 $\mathbf { x } _ { r 2 }$ 比第一个接收器更靠近外部硬壁边界，因此过早切割发生得更快。因此，图9(b)中的误差比图9(a)中的误差大。

Fig. 9 displays the dissipation error ?? and phase error ?? of the simulation results for both porous absorbers, which have markedly $\epsilon _ { a m p }$ $\epsilon _ { \vartheta }$ different values of flow resistivity $\sigma .$ . The combination of the mesh and the polynomial order results in spatial resolutions of 9 degrees of freedom per wavelength at 2. kHz. As expected. the numerical error maintains at a acceptable level when the spatial resolution is sufficient. It should be noted that the error in the low frequency range (below 500 Hz) is larger than the middle frequency range (500 − 1500 Hz). This is due to the premature cut of the recorded pressure signal, which leads to a loss of low-frequency power of the reflected sound field. Since the second receiver $\mathbf { x } _ { r 2 }$ of the oblique incidence case is closer to the outer hard wall boundary than the first receiver, the premature cut happens sooner. Consequently, the error in Fig. 9(b) is larger the one in Fig. 9(a).

## 5. 结论

## 5. Conclusions

在这项工作中，为了时域室内声学模拟的目的，开发了一个用于模拟多孔材料（包括薄覆盖物）的一般扩展反应边界的数值框架。等效流体模型用于描述任意多孔材料内的声波传播，其有效密度和压缩率可以使用多极有理函数很好地近似。通过应用ADE方法计算卷积积分，所得到的多孔材料时域控制方程可以写成统一的双曲线形式，就像无损空气的线性声学方程一样。基于解决潜在的黎曼问题，开发了一致的迎风数值通量公式，确保传播介质（包括空气、覆盖材料和多孔吸收体）之间适当的物理耦合。柔软渗透膜模型用于表征覆盖材料的声学特性。为了解决由于时间步长受限而导致的电位低效问题，采用了局部时间步长方案

In this work, a numerical framework for modeling general extended reacting boundaries of porous materials including thin coverings has been developed for the purpose of time-domain room acoustic simulations. The equivalent fluid models are used to describe the acoustic wave propagation inside arbitrary porous materials, the effective density and compressibility of which are well approximated using multi-pole rational functions. By applying the ADE method to calculate the convolution integral, the resulting time-domain governing equations of porous materials can be written in a unified hyperbolic form like the linear acoustic equations for lossless air. Based on solving the underlying Riemann problem, a consistent upwind numerical flux formulation that ensures appropriate physical coupling between propagation media, including air, covering materials, and porous absorbers, is developed. The limp permeable membrane model is used to characterize the acoustic properties of the covering materials. To tackle potentia inefficiency issues due to the constrained time step size, the local time-stepping scheme is employed

一维数值测试验证了所提出公式的收敛性，其中获得了 DG 方法的最佳收敛速率 $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$。同时，证明了接口耦合不会产生额外的误差。三维测试进一步验证了所提出的方法在多维情况下表示实际扩展反应阻抗边界的能力。幅度和相位信息均被准确捕获。

One-dimensional numerical tests verify the convergence property of the proposed formulation, where the optimal rate of convergence $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ for the DG method is obtained. Meanwhile, it is demonstrated that the interface coupling does not incur extra error. Three-dimensional tests further validate the capacity of the proposed methodology for representing practical extended reacting impedance boundaries in the multi-dimensional case. Both the magnitude and the phase information are accurately captured.

所提出的数值框架不仅提高了时域间断伽辽金方法在室内声学建模中的适用性，而且还可以应用于其他基于波的方法，例如有限体积方法。未来的工作将研究当前框架的扩展，以描述其他类型的边界材料，例如超材料。更复杂的表面覆盖模型的实现是另一个潜在的研究主题。通过采用所提出的模型为复杂的室内声学问题提供参考解决方案，可以探索和测试计算效率更高的扩展反应边界的替代模型。

The proposed numerical framework not only improves the applicability of the time-domain discontinuous Galerkin method for modeling room acoustics, but also can be applied to other wave-based methods, such as the finite volume method. Future work will investigate extension of the current framework to describe additional types of boundary materials, such as metamaterials. The implementation of more complicated surface covering models is an additional potential study topic. By employing the proposed model to provide reference solutions to complex room acoustic problems, it is possible to explore and test surrogate models of extended reacting boundaries that are more computationally efficient.

## CRediT 作者贡献声明

## CRediT authorship contribution statement

王惠清：概念化、方法论、软件、验证、形式分析、写作-初稿、写作-审查和编辑。 Maarten Hornikx：监督、写作审查和编辑、资金收购。

Huiqing Wang: Conceptualization, Methodology, Software, Validation, Formal analysis, Writing – original draft, Writing – review & editing. Maarten Hornikx: Supervision, Writing – review & editing, Funding acquisition.

## 数据可用性

## Data availability

文章中描述的研究没有使用任何数据。

No data was used for the research described in the article.

## 致谢

## Acknowledgments

这项工作由荷兰研究委员会 (NWO) 资助，拨款号为 19430。

This work is funded by the Dutch Research Council (NWO) under grant No. 19430.

## 参考文献

## References



[11 J. Allard、N. Atalla，多孔介质中声音的传播：吸声材料建模，John Wiley & Sons，2009 年。

[11 J. Allard, N. Atalla, Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials, John Wiley & Sons, 2009.





[2] L. Savioja，U.P. Svensson，几何房间声学建模技术概述，J. Acoust。苏克。是。 138（2）（2015）708-730。

[2] L. Savioja, U.P. Svensson, Overview of geometrical room acoustic modeling techniques, J. Acoust. Soc. Am. 138 (2) (2015) 708–730.





[3] T. Sakuma、S. Sakamoto、T. Otsuru，建筑与环境声学的计算模拟，Springer，2014 年。

[3] T. Sakuma, S. Sakamoto, T. Otsuru, Computational Simulation in Architectural and Environmental Acoustics, Springer, 2014.





[4] E. Brandão、A. Lenzi、S. Paul，原位阻抗和吸声测量技术综述，Acta Acust。联合阿科斯特。 101（3）（2015）443-463

[4] E. Brandão, A. Lenzi, S. Paul, A review of the in situ impedance and sound absorption measurement techniques, Acta Acust. United Acust. 101 (3) (2015) 443-463





[5] A. Southern、S. Siltanen、D.T. Murphy、L. Savioja，使用混合声学模型的房间脉冲响应合成和验证，IEEE Trans。音频、语音、语言。过程。 21（9）（2013）1940-1952。

[5] A. Southern, S. Siltanen, D.T. Murphy, L. Savioja, Room impulse response synthesis and validation using a hybrid acoustic model, IEEE Trans. Audio, Speech, Lang. Process. 21 (9) (2013) 1940–1952.





[6] S. Bilbao、B. Hamilton、J. Botts、L. Savioja，一般阻抗边界条件下的有限体积时域室内声学仿真，IEEE/ACM Trans。声音的。言语朗，过程。 （TASLP）24（1）（2016）161-173。

[6] S. Bilbao, B. Hamilton, J. Botts, L. Savioja, Finite volume time domain room acoustics simulation under general impedance boundary conditions, IEEE/ACM Trans. Audio. Speech Lang, Process. (TASLP) 24 (1) (2016) 161–173.





[7] H. Wang, M. Hornikx，使用不连续 Galerkin 方法进行室内声学模拟的时域阻抗边界条件建模，J. Acoust, Soc, Am。 147（4）（2020）2534–2546。

[7] H. Wang, M. Hornikx, Time-domain impedance boundary condition modeling with the discontinuous Galerkin method for room acoustics simulations, J. Acoust, Soc, Am. 147 (4) (2020) 2534–2546.





[8] E.平德。 A.P. Engsig-Karup。 C.-H。郑，LS.赫斯特文，M.S.美玲. J. Strømann-Andersen，使用谱元方法进行时域室内声学模拟，J. Acoust。苏克。是。 145（6）（2019）3299-3310。

[8] E. Pind. A.P. Engsig-Karup. C.-H. Jeong, LS. Hesthaven, M.S. Meiling. J. Strømann-Andersen, Time domain room acoustic simulations using the spectra element method, J. Acoust. Soc. Am. 145 (6) (2019) 3299–3310.





[9] T. Okuzono、T. Yoshida、K. Sakagami，使用时域 FEM 进行室内声学模拟（包括频率相关吸收边界条件）的效率：与频域 FEM 的比较，Appl。声学。 182（2021）108212。

[9] T. Okuzono, T. Yoshida, K. Sakagami, Efficiency of room acoustic simulations with time-domain FEM including frequency-dependent absorbing boundary conditions: Comparison with frequency-domain FEM, Appl. Acoust. 182 (2021) 108212.





[10] D.K.威尔逊，S.L.科利尔，V.E.奥斯塔舍夫，D.F.奥尔德里奇，N.P. Symons, D.H. Marlin，多孔表面声阻抗的时域建模 Acta Acust，United Acust，92 (6) (2006) 965–975。

[10] D.K. Wilson, S.L. Collier, V.E. Ostashev, D.F. Aldridge, N.P. Symons, D.H. Marlin, Time-domain modeling of the acoustic impedance of porous surfaces Acta Acust, United Acust, 92 (6) (2006) 965–975.





[11] V.E.奥斯塔舍夫，S.L.科利尔，D.K.威尔逊，D.F.奥尔德里奇，N.P. Symons，D. Marlin，多孔表面时域边界条件中的 Padé 近似，J. Acoust。苏克。是。 122（1）（2007）107-112。

[11] V.E. Ostashev, S.L. Collier, D.K. Wilson, D.F. Aldridge, N.P. Symons, D. Marlin, Padé approximation in time-domain boundary conditions of porous surfaces, J. Acoust. Soc. Am. 122 (1) (2007) 107–112.





[12] 下午莫尔斯，K.U. Ingard，《理论声学》，普林斯顿大学出版社，1986 年。

[12] P.M. Morse, K.U. Ingard, Theoretical Acoustics, Princeton University Press, 1986.





[13] C.-H。 Jeong，根据随机入射吸收系数采用多孔吸收体局部反应假设的指南，Acta Acust。联合阿库斯特，97 (5) (2011) 779–790。

[13] C.-H. Jeong, Guideline for adopting the local reaction assumption for porous absorbers in terms of random incidence absorption coefficients, Acta Acust. United Acust, 97 (5) (2011) 779–790.





[14] R. Dragonetti，R.A.罗马诺，非局部反应多孔层吸声的考虑，应用。声学。 87（2015）46-56。

[14] R. Dragonetti, R.A. Romano, Considerations on the sound absorption of non locally reacting porous layers, Appl. Acoust. 87 (2015) 46–56.





[15] R. Dragonetti，R.A. Romano，在表面声阻抗估计中假设局部反应边界条件时的错误，应用。声学。 115（2017）121-130。

[15] R. Dragonetti, R.A. Romano, Errors when assuming locally reacting boundary condition in the estimation of the surface acoustic impedance, Appl. Acoust. 115 (2017) 121–130.





[16] Y.高桥。 T. Otsuru，R. Tomiku，使用两个麦克风和环境噪声原位测量多孔材料的表面阻抗和吸收系数，Appl。声学。 66（7）（2005）845-865。

[16] Y. Takahashi. T. Otsuru, R. Tomiku, In situ measurements of surface impedance and absorption coefficients of porous materials using two microphones and ambient noise, Appl. Acoust. 66 (7) (2005) 845–865.





[17] R. Tomiku、T. Otsuru、N. Okamoto、T. Okuzono、T. Shibata，使用整体平均表面法向阻抗在混响室中进行有限元声场分析，见：INTER-NOISE 和 NOISE-CON 国会和会议论文集，卷。 2011，噪声控制工程研究所，2011，第1780-1785页

[17] R. Tomiku, T. Otsuru, N. Okamoto, T. Okuzono, T. Shibata, Finite element sound field analysis in a reverberation room using ensemble averaged surface normal impedance, in: INTER-NOISE and NOISE-CON Congress and Conference Proceedings, Vol. 2011, Institute of Noise Control Engineering, 2011, pp. 1780-1785





[18] M. Aretz、M. Vorländer，室内声学有限元模拟中吸收边界的高效建模，Acta Acust。联合阿科斯特。 96（6）（2010）1042-1050。

[18] M. Aretz, M. Vorländer, Efficient modelling of absorbing boundaries in room acoustic FE simulations, Acta Acust. United Acust. 96 (6) (2010) 1042–1050.





[19] Y. Yasuda，S. Ueno，M. Kadota，H. Sekine，局部反应边界条件对刚性壁支持的多孔材料层的适用性：具有不均匀分布的吸声表面的非扩散声场的波基数值研究，Appl。声学。 113（2016）45-57。

[19] Y. Yasuda, S. Ueno, M. Kadota, H. Sekine, Applicability of locally reacting boundary conditions to porous material layer backed by rigid wall: Wave-bas numerical study in non-diffuse sound field with unevenly distributed sound absorbing surfaces, Appl. Acoust. 113 (2016) 45–57.





[20] M. Hodgson、A. Wareing，具有扩展和局部反应边界表面的房间中预测稳态水平的比较，J. Sound Vib。 30（1-2）（2008）167-177。

[20] M. Hodgson, A. Wareing, Comparisons of predicted steady-state levels in rooms with extended-and local-reaction bounding surfaces, J. Sound Vib. 30 (1–2) (2008) 167–177.





[21] B. Yousefzadeh、M. Hodgson，使用不同边界条件对房间声学参数进行基于能量和波的波束跟踪预测，J. Acoust。苏克。是。 132（3）（2012）1450-1461。

[21] B. Yousefzadeh, M. Hodgson, Energy-and wave-based beam-tracing prediction of room-acoustical parameters using different boundary conditions, J. Acoust. Soc. Am. 132 (3) (2012) 1450–1461.





[22] K. Gunnarsdóttir，C.-H。 Jeong，G. Marbjerg，基于局部和扩展反应的多孔天花板吸声器的声学行为，J. Acoust。苏克。是。 137（1）（2015）509-512。

[22] K. Gunnarsdóttir, C.-H. Jeong, G. Marbjerg, Acoustic behavior of porous ceiling absorbers based on local and extended reaction, J. Acoust. Soc. Am. 137 (1) (2015) 509–512.





[23] M.A. Biot，多孔介质中的变形和声传播力学，J. Appl。物理。 33（4）（1962）1482-1498。

[23] M.A. Biot, Mechanics of deformation and acoustic propagation in porous media, J. Appl. Phys. 33 (4) (1962) 1482–1498.





[24] M.A. Biot，多孔耗散介质中声传播的广义理论，J. Acoust。苏克。是。 34（9）（1962）1254-1264。

[24] M.A. Biot, Generalized theory of acoustic propagation in porous dissipative media, J. Acoust. Soc. Am. 34 (9) (1962) 1254–1264.





[25] E. Deckers，N.-E。 Hörlin、D. Vandepitte、W. Desmet，基于波的二维多孔弹性 Biot 方程有效求解方法，Comput。方法应用机甲。工程。 201（2012）245-262。

[25] E. Deckers, N.-E. Hörlin, D. Vandepitte, W. Desmet, A Wave Based Method for the efficient solution of the 2D poroelastic Biot equations, Comput. Methods Appl. Mech. Engrg. 201 (2012) 245–262.





[26] J.-D。 Chazot，E. Perrey-Debain，B. Nennig，空气和多孔弹性介质中波模拟的统一有限元法的划分，J. Acoust。苏克。是。 135（2）（2014）724-733。

[26] J.-D. Chazot, E. Perrey-Debain, B. Nennig, The partition of unity finite element method for the simulation of waves in air and poroelastic media, J. Acoust. Soc. Am. 135 (2) (2014) 724–733.





[27] Y. Miki，多孔材料的声学特性 - Delany-Bazley 模型的修改，J. Acoust。苏克。日本（E）11（1）（1990）19-24。

[27] Y. Miki, Acoustical properties of porous materials-modifications of Delany-Bazley models, J. Acoust. Soc. Japan (E) 11 (1) (1990) 19–24.





[28] J.-F。 Allard，Y. Champoux，刚性框架纤维材料中声音传播的新经验方程，J. Acoust。苏克。是。 91（6）（1992）3346–3353

[28] J.-F. Allard, Y. Champoux, New empirical equations for sound propagation in rigid frame fibrous materials, J. Acoust. Soc. Am. 91 (6) (1992) 3346–3353





[29] D. Lafarge、P. Lemarinier、J.F. Allard、V. Tarnow，可听频率下多孔结构中空气的动态压缩性，J. Acoust。苏克。是。 102（4）（1997）1995-2006。

[29] D. Lafarge, P. Lemarinier, J.F. Allard, V. Tarnow, Dynamic compressibility of air in porous structures at audible frequencies, J. Acoust. Soc. Am. 102 (4) (1997) 1995–2006.





[30] D.K.威尔逊，V.E.奥斯塔舍夫，S.L.科利尔，N.P.西蒙斯，D.F. Aldridge，D.H. Marlin，声音与室外地面相互作用的时域计算，Appl。声学。 68（2）（2007）173-200。

[30] D.K. Wilson, V.E. Ostashev, S.L. Collier, N.P. Symons, D.F. Aldridge, D.H. Marlin, Time-domain calculations of sound interactions with outdoor ground surfaces, Appl. Acoust. 68 (2) (2007) 173–200.





[31] M.费拉，Z.E.A. Fellah、E. Ogam、F. Mitri、C. Dépollier，低频连续非均匀刚性框架多孔材料瞬态波传播的广义方程，J. Acoust。苏克。是。 134（6）（2013）4642-4647。

[31] M. Fellah, Z.E.A. Fellah, E. Ogam, F. Mitri, C. Dépollier, Generalized equation for transient-wave propagation in continuous inhomogeneous rigid-frame porous materials at low frequencies, J. Acoust. Soc. Am. 134 (6) (2013) 4642–4647.





[32] D.K.威尔逊，V.E.奥斯塔舍夫，S.L. Collier，刚性多孔介质中声音传播的时域方程，J. Acoust。苏克。是。 116（4）（2004）1889-1892。

[32] D.K. Wilson, V.E. Ostashev, S.L. Collier, Time-domain equations for sound propagation in rigid-frame porous media, J. Acoust. Soc. Am. 116 (4) (2004) 1889–1892.





[33] O. Umnova、D. Turo，刚性多孔介质等效流体模型的时域公式，J. Acoust。苏克。是。 125（4）（2009）1860–1863

[33] O. Umnova, D. Turo, Time domain formulation of the equivalent fluid model for rigid porous media, J. Acoust. Soc. Am. 125 (4) (2009) 1860–1863





[34] J. 赵，M. 包，X. Wang，H. Lee，S. Sakamoto，基于等效流体模型的刚性框架多孔材料中声音传播的有限差分时域算法，J. Acoust。苏克。是。 143（1）（2018）130-138。

[34] J. Zhao, M. Bao, X. Wang, H. Lee, S. Sakamoto, An equivalent fluid model based finite-difference time-domain algorithm for sound propagation in porous material with rigid frame, J. Acoust. Soc. Am. 143 (1) (2018) 130–138.





[35] D. Dragna、P. Pineau、P. Blanc-Benon，多孔介质中时域传播的广义递归卷积方法，J. Acoust。苏克。是。 138（2）（2015）1030-1042。

[35] D. Dragna, P. Pineau, P. Blanc-Benon, A generalized recursive convolution method for time-domain propagation in porous media, J. Acoust. Soc. Am. 138 (2) (2015) 1030–1042.





[36] I. Moufid、D. Matignon、R. Roncen、E. Piot，刚性多孔介质中波传播时域等效流体模型的能量分析和离散化，J. Comput。物理。 451（2022）110888。

[36] I. Moufid, D. Matignon, R. Roncen, E. Piot, Energy analysis and discretization of the time-domain equivalent fluid model for wave propagation in rigid porous media, J. Comput. Phys. 451 (2022) 110888.





[37] F. Pind，C.-H。 Jeong, A.P. Engsig-Karup, J.S. Hesthaven，J. Strømann-Andersen，使用不连续 Galerkin 方法对扩展反应多孔吸声体进行时域室内声学模拟，J. Acoust。苏克。是。 148（5）（2020）2851-2863。

[37] F. Pind, C.-H. Jeong, A.P. Engsig-Karup, J.S. Hesthaven, J. Strømann-Andersen, Time-domain room acoustic simulations with extended-reacting porous absorbers using the discontinuous Galerkin method, J. Acoust. Soc. Am. 148 (5) (2020) 2851–2863.





[38] T. Yoshida、T. Okuzono、K. Sakagami，基于等效流体模型的多孔吸声体时域有限元公式，Acoust。科学。技术。 41（6）（2020）837-840。

[38] T. Yoshida, T. Okuzono, K. Sakagami, Time-domain finite element formulation of porous sound absorbers based on an equivalent fluid model, Acoust. Sci. Technol. 41 (6) (2020) 837–840.





[39] A.阿洛马尔，D.德拉格纳，M.-A。 Galland，带有扩展反应衬里的流道中声音传播的时域模拟，J. Sound Vib。 507（2021）116137。

[39] A. Alomar, D. Dragna, M.-A. Galland, Time-domain simulations of sound propagation in a flow duct with extended-reacting liners, J. Sound Vib. 507 (2021) 116137.





[40] 谢明宇Ou，L. Xu，具有记忆项的正交各向异性多孔弹性介质中波传播的不连续伽辽金方法，J. Comput。物理。 397（2019）108865。

[40] J. Xie, M.Y. Ou, L. Xu, A discontinuous Galerkin method for wave propagation in orthotropic poroelastic media with memory terms, J. Comput. Phys. 397 (2019) 108865.





[41] T. Bravo，C. Maury，各向异性纤维材料支持的微穿孔板的声音衰减和吸收：理论和实验研究，J. Sound Vib。 425（2018）189-207。

[41] T. Bravo, C. Maury, Sound attenuation and absorption by micro-perforated panels backed by anisotropic fibrous materials: Theoretical and experimental study, J. Sound Vib. 425 (2018) 189–207.





[42] K. Sakagami、S. Kobatake、K. Kano、M. Morimoto、M. Yairi，多孔吸收层支持的单个微穿孔面板吸声器的吸声特性，Acoust。澳大利亚 39 (3) (2011)。

[42] K. Sakagami, S. Kobatake, K. Kano, M. Morimoto, M. Yairi, Sound absorption characteristics of a single microperforated panel absorber backed by a porous absorbent layer, Acoust. Australia 39 (3) (2011).





[43] T. Okuzono，K. Uenishi，K. Sakagami，不同背衬气腔设计的单叶渗透膜吸收体吸收特性的实验比较，噪声控制工程，J. 68（3）（2020）237-245。

[43] T. Okuzono, K. Uenishi, K. Sakagami, Experimental comparison of absorption characteristics of single-leaf permeable membrane absorbers with different backing air cavity designs, Noise Control Eng, J. 68 (3) (2020) 237–245.





[44] M. Toyoda，J. Motooka，使用时域有限差分法预测可渗透薄吸收体，J. Acoust。苏克。是。 143（5）（2018）2870-2877

[44] M. Toyoda, J. Motooka, Prediction of permeable thin absorbers using the finite-difference time-domain method, J. Acoust. Soc. Am. 143 (5) (2018) 2870-2877





[45] T. Okuzono，N. Shimizu，K. Sakagami，利用时域有限元法预测单叶渗透膜吸收器的吸收特性，Appl.，Acoust，151（2019）172-182

[45] T. Okuzono, N. Shimizu, K. Sakagami, Predicting absorption characteristics of single-leaf permeable membrane absorbers using finite element method in a time domain, Appl., Acoust, 151 (2019) 172–182





[46] S. Mukae，T. Okuzono，K. Tamaru。 K. Sakagami，利用平面波富集有限元对室内声学解算器的微穿孔板和渗透膜进行建模，Appl。声学。 185（2022）108383。

[46] S. Mukae, T. Okuzono, K. Tamaru. K. Sakagami, Modeling microperforated panels and permeable membranes for a room acoustic solver with plane-wave enriched fem, Appl. Acoust. 185 (2022) 108383.





[47] T. Okuzono、K. Sakagami，用于具有微孔面板吸声器的 3D 房间声学模拟的频域有限元求解器，Appl Acoust。 129（2018）1-12。

[47] T. Okuzono, K. Sakagami, A frequency domain finite element solver for acoustic simulations of 3D rooms with microperforated panel absorbers, Appl Acoust. 129 (2018) 1–12.





[48] M. Toyoda，D. Eto，使用时域有限差分法预测微穿孔板吸波器，Wave Motion 86 (2019) 110-124。

[48] M. Toyoda, D. Eto, Prediction of microperforated panel absorbers using the finite-difference time-domain method, Wave Motion 86 (2019) 110–124.





[49] T. Wu, C. Cheng, Z. Tao. 带保护布和嵌入薄表面的填充消音器的边界元分析，J. Sound Vib, 261 (1) (2003 1-15.

[49] T. Wu, C. Cheng, Z. Tao, Boundary element analysis of packed silencers with protective cloth and embedded thin surfaces, J. Sound Vib, 261 (1) (2003 1–15.





[50] G. Gabard，O. Dazel，用于吸声材料的平面波不连续伽辽金方法，Internat。 J. 数字。方法工程。 104（12）（2015）1115-1138。

[50] G. Gabard, O. Dazel, A discontinuous Galerkin method with plane waves for sound-absorbing materials, Internat. J. Numer. Methods Engrg. 104 (12) (2015) 1115–1138.





[51] S. Wu，O. Dazel，G. Gabard，G. Legrain，用于模拟具有耦合界面的吸声多孔弹性材料的高阶 X-FEM，J. Sound Vib。 510（2021）116262。

[51] S. Wu, O. Dazel, G. Gabard, G. Legrain, High-order X-FEM for the simulation of sound absorbing poro-elastic materials with coupling interfaces, J. Sound Vib. 510 (2021) 116262.





[52] J.Zhao，Z.Chen，M.Bao，S.Sakamoto，利用有限差分时域分析预测声学楔形吸声系数，应用。声学。 155（2019）428-441。

[52] J. Zhao, Z. Chen, M. Bao, S. Sakamoto, Prediction of sound absorption coefficients of acoustic wedges using finite-difference time-domain analysis, Appl. Acoust. 155 (2019) 428–441.





[53] A.阿洛马尔，D.德拉格纳，M.-A。 Galland，提取一般吸声材料等效流体特性的极识别方法，Appl。声学。 174（2021）107752。

[53] A. Alomar, D. Dragna, M.-A. Galland, Pole identification method to extract the equivalent fluid characteristics of general sound-absorbing materials, Appl. Acoust. 174 (2021) 107752.





[54] B. Gustaysen、A. Semlven，通过矢量拟合对频域响应进行有理逼近，IEEE Trans。电力输送14（3）（1999）1052-1061。

[54] B. Gustaysen, A. Semlven, Rational approximation of frequency domain responses by vector fitting, IEEE Trans. Power Deliv. 14 (3) (1999) 1052–1061.





[55] R.M. Joseph，S.C. Hagness，A. Taflove，线性色散介质中麦克斯韦方程的直接时间积分，吸收飞秒电磁脉冲的散射和传播，选项。莱特。 16（18）（1991）1412-1414。

[55] R.M. Joseph, S.C. Hagness, A. Taflove, Direct time integration of Maxwell’s equations in linear dispersive media with absorption for scattering and propagation of femtosecond electromagnetic pulses, Opt. Lett. 16 (18) (1991) 1412–1414.





[56] A.D. Pierce，声学：物理原理和应用简介，Springer International Publishing，Springer Nature，Switzerland AG 2019。

[56] A.D. Pierce, Acoustics: An Introduction to Its Physical Principles and Applications, Springer International Publishing, Springer Nature, Switzerland AG 2019.





[57] J.S. Hesthaven, T. Warburton，节点不连续伽辽金方法：算法、分析和应用，Springer-Verlag，纽约，2007 年。

[57] J.S. Hesthaven, T. Warburton, Nodal Discontinuous Galerkin Methods: Algorithms, Analysis and Applications, Springer-Verlag, New York, 2007.





[58] H. Wang、I. Sihar、R. Pagán Muñoz、M. Hornikx，使用节点不连续 Galerkin 方法进行时域室内声学建模，J. Acoust。苏克。是。 145（4）（2019）2650–2663。

[58] H. Wang, I. Sihar, R. Pagán Muñoz, M. Hornikx, Room acoustics modelling in the time-domain with the nodal discontinuous Galerkin method, J. Acoust. Soc. Am. 145 (4) (2019) 2650–2663.





[59] Shu C.-W.，用于时间相关问题的不连续伽辽金方法：调查和最新进展，载于：偏微分方程的不连续伽辽金有限元方法的最新发展，2014，第 25-62 页。

[59] Shu C.-W., Discontinuous Galerkin method for time-dependent problems: Survey and recent developments, in: Recent Developments in Discontinuous Galerkin Finite Element Methods for Partial Differential Equations, 2014, pp. 25–62.





[60] D.A. Kopriva, J. Nordström, G.J.加斯纳，双曲问题的不连续伽辽金谱元近似的误差有界，J. Sci。计算。 72（1）（2017）314-330。

[60] D.A. Kopriva, J. Nordström, G.J. Gassner, Error boundedness of discontinuous Galerkin spectral element approximations of hyperbolic problems, J. Sci. Comput. 72 (1) (2017) 314–330.





[61] K. Duru，L. Rannabauer，A.-A。 Gabriel, H. Igel，一种新的非连续伽辽金方法，用于具有物理激励数值通量的弹性波，J. Sci。计算。 88（3）（2021）1-32。

[61] K. Duru, L. Rannabauer, A.-A. Gabriel, H. Igel, A new discontinuous Galerkin method for elastic waves with physically motivated numerical fluxes, J. Sci. Comput. 88 (3) (2021) 1–32.





[62] M. Ainsworth，高阶间断伽辽金有限元方法的色散和耗散行为，J. Comput。物理。 198（1）（2004）106–130

[62] M. Ainsworth, Dispersive and dissipative behaviour of high order discontinuous Galerkin finite element methods, J. Comput. Phys. 198 (1) (2004) 106–130





[63] R.J. LeVeque，双曲问题的有限体积方法，剑桥大学出版社，剑桥，2002 年

[63] R.J. LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, Cambridge, 2002





[64] L.C. Wilcox、G. Stadler、C. Burstedde、O. Ghattas，一种通过弹性声学耦合介质传播波的高阶不连续伽辽金方法，J. Comput。物理。 229 (24) (2010) 9373–9396。

[64] L.C. Wilcox, G. Stadler, C. Burstedde, O. Ghattas, A high-order discontinuous Galerkin method for wave propagation through coupled elastic–acoustic media, J. Comput. Phys. 229 (24) (2010) 9373–9396.





[65] 詹强，庄明，方勇，胡勇，毛勇，W.-F.黄荣、张、王大、Q.H.刘，全各向异性孔隙弹性波建模：具有广义波阻抗的不连续伽辽金算法，Comput。方法应用机甲。工程。 346（2019）288-311。

[65] Q. Zhan, M. Zhuang, Y. Fang, Y. Hu, Y. Mao, W.-F. Huang, R. Zhang, D. Wang, Q.H. Liu, Full-anisotropic poroelastic wave modeling: A discontinuous Galerkin algorithm with a generalized wave impedance, Comput. Methods Appl. Mech. Engrg. 346 (2019) 288–311.





[66] H.-O。 Kreiss，双曲系统的初始边值问题，Comm。纯应用。数学。 23（3）（1970）277-298。

[66] H.-O. Kreiss, Initial boundary value problems for hyperbolic systems, Comm. Pure Appl. Math. 23 (3) (1970) 277–298.





[67] A. Majda，S. Osher，具有均匀特征边界的双曲方程的初始边值问题，Comm。纯应用。数学。 28（5）（1975）607-675。

[67] A. Majda, S. Osher, Initial–boundary value problems for hyperbolic equations with uniformly characteristic boundary, Comm. Pure Appl. Math. 28 (5) (1975) 607–675.





[68] R.L. Higdon，线性双曲系统的初始边值问题，SIAM Rev. 28 (2) (1986) 177-217。

[68] R.L. Higdon, Initial–boundary value problems for linear hyperbolic system, SIAM Rev. 28 (2) (1986) 177–217.





[69] H. Wang，J. Yang，M. Hornikx，声学时域节点间断伽辽金模型中的频率相关传输边界条件，应用。声学。 164（2020）107280。

[69] H. Wang, J. Yang, M. Hornikx, Frequency-dependent transmission boundary condition in the acoustic time-domain nodal discontinuous Galerkin model, Appl. Acoust. 164 (2020) 107280.





[70] L.M. Brekhoyskikh，O.A. Godin，Lavered Media 声学 I：平面波和 Ouasi 平面波，卷。 5.施普林格科学与商业媒体，2012

[70] L.M. Brekhoyskikh, O.A. Godin, Acoustics of Lavered Media I: Plane and Ouasi-Plane Waves, Vol. 5. Springer Science & Business Media, 2012





[71]D.-Y。 Maa，微孔板吸波器的潜力，J. Acoust。苏克。是。 104（5）（1998）2861-2866。

[71] D.-Y. Maa, Potential of microperforated panel absorber, J. Acoust. Soc. Am. 104 (5) (1998) 2861–2866.





[72] H. Wang、M. Cosnefroy、M. Hornikx，用于线性声波传播的具有局部时间步进的任意高阶不连续 Galerkin 方法，J. Acoust。苏克。是。 149（1）（2021）569-580。

[72] H. Wang, M. Cosnefroy, M. Hornikx, An arbitrary high-order discontinuous Galerkin method with local time-stepping for linear acoustic wave propagation, J. Acoust. Soc. Am. 149 (1) (2021) 569–580.





[73] B.Cockburn，C.-W。 Shu，Runge-Kutta 不连续 Galerkin 方法解决对流主导问题，J. Sci。计算机，16 (3) (2001) 173–261

[73] B. Cockburn, C.-W. Shu, Runge-Kutta discontinuous Galerkin methods for convection-dominated problems, J. Sci. Comput, 16 (3) (2001) 173–261





[74] B. Cotté、P. Blanc-Benon、C. Bogey、F. Poisson，室外声音传播模拟的时域阻抗边界条件，AIAA J. 47 (10) (2009) 2391–2403。

[74] B. Cotté, P. Blanc-Benon, C. Bogey, F. Poisson, Time-domain impedance boundary conditions for simulations of outdoor sound propagation, AIAA J. 47 (10) (2009) 2391–2403.





[75] T. Toulorge、W. Desmet，应用于波传播问题的不连续伽辽金空间离散化的最佳龙格-库塔方案，J. Comput。物理。 231（4）（2012）2067-2091。

[75] T. Toulorge, W. Desmet, Optimal Runge–Kutta schemes for discontinuous Galerkin space discretizations applied to wave propagation problems, J. Comput. Phys. 231 (4) (2012) 2067–2091.





[76] J.-F。 Allard、W. Lauriks、C. Verhaegen，多孔层上方的声场以及根据自由场测量估计声表面阻抗，J. Acoust。苏克。是。 91（5）（1992）3057-3060。

[76] J.-F. Allard, W. Lauriks, C. Verhaegen, The acoustic sound field above a porous layer and the estimation of the acoustic surface impedance from free-field measurements, J. Acoust. Soc. Am. 91 (5) (1992) 3057–3060.





[77] C. Geuzaine，J.-F。 Remacle，Gmsh：具有内置预处理和后处理设施的 3-D 有限元网格生成器，Internat。 J. 数字。方法工程。 79（11）（2009）1309-1331。

[77] C. Geuzaine, J.-F. Remacle, Gmsh: A 3-D finite element mesh generator with built-in pre-and post-processing facilities, Internat. J. Numer. Methods Engrg. 79 (11) (2009) 1309–1331.

