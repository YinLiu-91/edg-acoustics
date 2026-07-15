# Extended reacting boundary modeling of porous materials with thin coverings for time-domain room acoustic simulations

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/61067af0e10238ed13ffe85864db96029c4beddd791f2d1d0ccc21153ff064a2.jpg)


Huiqing Wang <sup>∗</sup>, Maarten Hornikx 

Building Acoustics, Department of the Built Environment, Eindhoven University of Technology, P.O. Box 513, 5600 MB Eindhoven, The Netherlands 

A R T I C L E I N F O 

Keywords: Time-domain room acoustic modeling Extended reaction of porous layers Thin covering materials High-order accuracy Exact Riemann solver 

## A B S T R A C T

Modeling of acoustic boundary conditions has a significant impact on the accuracy of room acoustic simulations, which play an important role in the design phase of indoor built environments in order to improve the acoustical comfort. In this work, a numerical framework based on the discontinuous Galerkin (DG) method is presented for modeling extended reacting boundaries of porous absorbers covered by thin materials. The domain decomposition methodology is applied by treating the porous material as a subdomain. Equivalent fluid models are used to depict the acoustic properties of porous materials, whose effective density and compressibility as irrational functions are approximated by multipole rational functions in the frequency domain. By employing the auxiliary differential equation approach to calculate the time convolution, the augmented time-domain governing equations of porous materials can be expressed in the same unified hyperbolic form as the linear acoustic equations, which further enables a consistent upwind numerical flux formulation throughout the whole domain. The numerical coupling across the interface between propagation media is handled by solving the underlying Riemann problem. Compared to existing approaches with the DG method for extended reacting boundaries modeling for room acoustics, the derived upwind numerical flux formulation does not involve the computation of auxiliary variables. The presented framework yield a wellposed linear hyperbolic system with admissible boundary conditions as guided by the ‘‘uniform Kreiss condition’’ (Kreiss, 1970). Acoustic properties of the covering materials are illustrated by considering a limp permeable membrane model. A local time-stepping approach is utilized to improve computational efficiency. Numerical validations against analytical solutions in 1D are performed to verify the desired high-order convergence rate. A 3D case study on modeling spherical wave fronts demonstrates the broadband accuracy of the formulation. 

## 1. Introduction

Porous materials are essential for a variety of applications in acoustics due to their broadband sound absorption capabilities and cost-effectiveness. For instance, they serve as necessary acoustic treatments of boundary surfaces in order to improve the acoustic comfort of built environments. Therefore, accurate modeling of sound wave propagation in the immediate vicinity of porous absorbers has been a subject of ongoing research [1]. In room acoustic modeling, acoustic absorption properties of porous materials are typically characterized by absorption coefficients for geometrical acoustic methods [2] or surface impedances for wave-based methods [3]. These properties fundamentally depend on the frequency and the angle of incidence waves. exhibiting the so-called extended reacting (ER) behavior that poses challenges for both rigorous numerical modeling [2] and impedances measurements [4]. To address this issue, a simplifying local reacting (LR) approximation is extensively assumed for both geometrical acoustic methods and wave-based methods [2,5–11], which asserts that the response at a certain point on the boundary surface is dependent only on the sound pressure incident on that specific location regardless of the surrounding acoustic field [12]. 

To elucidate the applicable range of the LR assumption, numerous numerical and experimental studies have been conducted. By investigating wave propagation models inside porous materials of varying degrees of complexity, it was found that the LR approximation results in more noticeable deviations in terms of predicted random incidence absorption coefficients as the flow resistivity decreases and the thickness of materials increases [13]. Later studies [14,15] based on both analytical models and numerical experiments further support these findings. Room acoustic simulations of a diffuse sound field using the frequency-domain finite element method [16–18] have shown that the ensemble-averaged surface impedance yields results that are comparably closer to reference values than those with the normal incidence impedance. For non-diffuse sound fields in general, applying LR and ER models reveals discrepancies in terms of reverberation decay curves and sound pressure level distributions, as analyzed by Yasuda et al. [19] with the multi-domain boundary element method. Numerical studies using the combined beam-tracing and transfer-matrix model showed that the LR model is highly inaccurate for surfaces consisting of multilayer porous materials [20,21]. Furthermore, it is well understood that ER and LR models behave considerably differently when the sound wave impinges on boundaries close to grazing incidence, as typically occurs around suspended ceilings [22]. Therefore, it is important to accomplish accurate modeling of ER porous materials for room acoustic analysis. 

To that end, a straightforward solution is to simulate wave propagation in all directions inside the material. The Biot poroelastic model [23,24] provides systematic and comprehensive descriptions of both the airborne acoustic wave and the frame-associated elastic waves. However, explicit calculations of all wave forms are computationally intensive [25,26]. Under the long-wavelength condition, i.e., the wavelength is significantly larger than microscopic sizes of typical pores, it is justifiable to view porous material with a rigid or limp frame as an equivalent fluid characterized by its effective density and bulk modulus on a macroscopic scale [1], which reduces the calculation loads remarkably. Therefore, our interest is restricted to porous materials that can be described by the equivalent fluid model (EFM). In the literature, EFMs, either empirical or phenomenological, are mostly formulated as irrationa transfer functions, e.g., the Miki model [27] and the Johnson–Champoux–Allard–Lafarge (JCAL) model [28,29]. To allow analytical time-domain formulations of the EFM, ad hoc assumptions are usually applied. However, some formulations [30,31] are valid to a limited frequency range, and other representations [32,33] involve convolution kernels with fractional terms that pose challenges in terms of numerical discretization and storage of solution history. 

Direct application of the inverse Fourier transform to the EFMs of irrational nature produces fractional differential operators in the time domain. To avoid that, existing numerical treatments of time-domain wave propagation in porous media mostly approximate the frequency-dependent attributes of EFMs in the frequency domain in the form of multipole rational functions, which are also known as IIR filters in signal processing. Zhao et al. [34] applied the ??-transform to the IIR filters and frequency-domain wave equations in order to avoid the convolution integrals in the time domain. Another approach is to numerically discretize the resulting time convolution integrals by the auxiliary differential equation (ADE) method. It differentiates the convolution integrals in time and transforms them into an additional set of first-order ordinary differential equations (ODEs) of auxiliary or memory variables, which can be solved with a high-order time integration scheme. The ADE method was applied to the Wilson’s relaxation EFM model [30] to simulate outdoor sound propagation by Dragna et al. [35]. More recently, a multipole based time-domain formulation of the EFM was presented by Moufid et al. [36] for wave propagation inside rigid porous media. Therein, a thorough energy and stability analysis showed that the positivity of fitting poles are necessary conditions to have stable solutions. More recent works relevant to the extended reaction modeling of porous materials using the ADE method include Refs. [37–39]. Yoshida et al. [38] presented an implicit time-domain finite element formulation for the scalar wave equation, where the auxiliary variables are discretized in the same manner as the primary acoustic variables. In [37], Pind et al. applied the multipole approximation to the Miki model for room acoustics, where the governing equations are spatially discretized by the discontinuous Galerkin method using the central flux. One issue with this work is that the auxiliary equations contain spatial derivatives of the principal acoustic variables, and consequently the computational cost increases with the number of auxiliary variables. Alomar et al. [39] circumvented this issue by using the partial fractional decomposition, similar to other works [36,40], and simulated acoustic propagation in flow ducts with extended-reacting liners using the finite difference scheme. 

In modern architectural designs, additional coverings made from fibers or perforated plates are typically attached to porous absorbers for purposes of hygiene, durability, and protection, Meanwhile, improved sound absorption performance in terms of the broadened absorption peak frequency range are observed [41–43]. Typical examples can be found in acoustic curtains and suspended ceilings. Extensive research efforts have been devoted to numerical modeling of sound absorption and transmission of these covering materials surrounded by air, such as permeable thin membranes [44–46] and microperforated panels [46–48], numerical schemes suitable for ER porous absorbers with covering materials are relatively rare to the authors' knowledge. Since the thickness of the cover is relatively small this thin covering cloth can be acoustically represented by its transfer impedance as a pressure jumr discontinuity instead of being modeled as another acoustic domain. This approach, which has been successfully applied in numerical modeling of mufflers [49] and flow ducts [39], is adopted in this work. Being a LR surface, the influence of coverings can be easily integrated into the impedance boundary formulation by augmenting the surface impedance of porous absorbers with its transfe impedance. In contrast, an ER boundary formulation necessitates an appropriate interface coupling condition between different propagation media, which was discussed extensively for the time-harmonic analysis of pore-elastic materials in Refs. [50,51]. However, a time-domain ER boundary formulation for interface coupling involving porous materials and thin coverings has not been developed. 

The main focus of this work is to present a time-domain formulation for extended reacting boundary of porous materials within the framework of the time-domain discontinuous Galerkin method. The proposed formulation serves room acoustic modeling purposes and handles porous absorbers that are either directly exposed or covered by thin surface coverings as in typical room acoustic scenarios. To fully represent the extended reaction of boundaries, acoustic wave propagation in all propagation media are simulated and coupled explicitly. The time-domain governing equations for the porous media domain are constructed by transforming EFMs based on the combination of rational approximations and the ADE method. This general formulation enables not only flexibilities in material models but also consistent and efficient numerical discretizations. The second contribution of this work is that an exact Riemann solver, which is a physically inspired numerical flux formulation, is developed for precise coupling between the propagation media with jump discontinuities. The rest of this paper is organized as follows. Section 2 presents the governing equations of sound propagation in air and porous materials in a unified hyperbolic form. Section 3 describes the numerical schemes. In Section $^ { 4 , }$ numerical verifications and applications are performed. Concluding remarks are given in Section 5. 

## 2. Time-domain formulations

For the purpose of clarity, a bounded domain $\varOmega ,$ composed of two homogeneous propagation media (the air and the porous material) with their respective physical governing equations and properties, is considered to illustrate the formulations in this work. This formulation can be extended to problems involving multiple propagation media of different properties straightforwardly. 

## 2.1. Air domain

Acoustic wave propagation in the motionless air subdomain $\varOmega _ { a }$ can be described by linear acoustic equations 

$$
\begin{array}{r} \frac {\partial \mathbf {v}}{\partial t} + \frac {1}{\rho_ {a}} \nabla p = \mathbf {0}, \\ \frac {\partial p}{\partial t} + \rho_ {a} c _ {a} ^ {2} \nabla \cdot \mathbf {v} = 0, \end{array}\tag{1}
$$

(2) 

where $\mathbf { v } ( \mathbf { x } , t )$ is the particle velocity vector with components $\{ v _ { x } , v _ { y } , v _ { z } \}$ , ??(??, ??) is the sound pressure, ?? denotes the spatial position, $\rho _ { a } = 1 . 2 ~ \mathrm { k g } / \mathrm { m } ^ { 3 }$ is the constant density of air and $c _ { a } = 3 4 3 ~ \mathrm { m / s }$ is the constant speed of sound. 

## 2.2. Porous material domain

The EFMs governing sound propagation inside the porous material subdomain $\varOmega _ { m }$ are expressed in the frequency-domain as (assuming $\mathrm { e } ^ { \mathrm { i } \omega t }$ time convention) 

$$
\begin{array}{r} \mathrm{i} \omega \rho_ {\mathrm{ef}} (\omega) \hat {\mathbf {v}} + \nabla \hat {p} = \mathbf {0}, \\ \mathrm{i} \omega \mathcal {C} _ {\mathrm{ef}} (\omega) \hat {p} + \nabla \cdot \hat {\mathbf {v}} = 0, \end{array}\tag{3a}
$$

(3b) 

where the hat notation (̂⋅) labels frequency-domain variables, $\rho _ { \mathrm { e f } } ( \omega )$ is the effective density, $c _ { \mathrm { e f } } ( \omega )$ is the effective compressibility (i.e. the inverse of effective bulk modulus) of the porous medium. To obtain time-domain governing equations, we first approximate the complex-valued $\rho _ { \mathrm { e f } } ( \omega )$ and $c _ { \mathrm { e f } } ( \omega )$ using rational functions with real poles as 

$$
\begin{array}{r} \rho_ {\mathrm{ef}} (\omega) \approx \rho_ {m} + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} \frac {B _ {\rho k}}{\zeta_ {\rho k} + \mathrm{i} \omega}, \\ \mathcal {C} _ {\mathrm{ef}} (\omega) \approx \mathcal {C} _ {m} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} \frac {B _ {C k}}{\zeta_ {C k} + \mathrm{i} \omega}, \end{array}\tag{4a}
$$

(4b) 

where $[ B _ { \rho k } , B _ { C k } ] \in \mathbb { R }$ and $[ \zeta _ { \rho k } , \zeta _ { C k } ] \in \mathbb { R } ^ { + }$ are fitting weights and poles respectively. $\rho _ { m }$ is the asymptotic value of effective density as the frequency approaches infinity, whereas $c _ { m }$ denotes the high-frequency asymptotic value of the effective compressibility. As both of them are constant, the asymptotic value of the sound speed in the material $c _ { m }$ is a constant that is equal to $1 / \sqrt { \rho _ { m } C _ { m } } .$ . Latest numerical studies $[ 3 6 , 3 9 , 5 2 ]$ and analysis [53] on EFM have demonstrated that rational functions with real poles are sufficient to capture the dissipative nature of conventional porous materials. In order to obtain stable solutions, all poles must be positive as proved in [36]. In this work, a vector fitting (VF) algorithm [54], which can impose this stability condition explicitly, is used for determining the fitting parameters due to its high accuracy and efficiency. By substituting Eqs. (4) into Eqs. (3) and applying the partial fraction decomposition to terms i?? $\rho _ { \mathrm { e f } }$ and i?? $c _ { \mathrm { e f } }$ therein, we get 

$$
\mathrm{i} \omega \rho_ {m} \hat {\mathbf {v}} + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} \big (B _ {\rho k} - \frac {B _ {\rho k} \zeta_ {\rho k}}{\zeta_ {\rho k} + \mathrm{i} \omega} \big) \hat {\mathbf {v}} + \nabla \hat {p} = \mathbf {0},\tag{5a}
$$

$$
\mathrm{i} \omega \mathcal {C} _ {m} \hat {p} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} \big (B _ {C k} - \frac {B _ {C k} \zeta_ {C k}}{\zeta_ {C k} + \mathrm{i} \omega} \big) \hat {p} + \nabla \cdot \hat {\mathbf {v}} = 0.\tag{5b}
$$

Then, applying the inverse Fourier transform and the auxiliary differential equations (ADE) method [35,55] results in 

$$
\begin{array}{r} \rho_ {m} \frac {\partial \mathbf {v}}{\partial t} + \nabla p + \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \mathbf {v} - \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \boldsymbol {\phi} _ {\rho k} = \mathbf {0}, \\ \frac {1}{\rho_ {m} c _ {m} ^ {2}} \frac {\partial p}{\partial t} + \nabla \cdot \mathbf {v} + \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} p - \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \zeta_ {C k} \boldsymbol {\phi} _ {C k} = 0, \end{array}\tag{6a}
$$

(6b) 

where $\boldsymbol { \phi } _ { \rho k } = [ \phi _ { \rho k } ^ { x } , \phi _ { \rho k } ^ { y } , \phi _ { \rho k } ^ { z } ] ^ { \mathrm { T } }$ and $\phi _ { C k }$ are the so-called accumulators or auxiliary variables that correspond to the convolution integral associated with the components of ?? and $p ,$ respectively, e.g., 

$$
\phi_ {\rho k} (\mathbf {x}, t) = \int_ {0} ^ {t} \mathbf {v} (\mathbf {x}, \tau) \mathrm{e} ^ {- \zeta_ {\rho k} (t - \tau)} \mathrm{d} \tau .
$$

They are governed by time-dependent ordinary differential equations 

$$
\begin{array}{r l} & {\frac {\partial \pmb {\phi} _ {\rho k}}{\partial t} + \zeta_ {\rho k} \pmb {\phi} _ {\rho k} = \mathbf {v}, \quad \forall k \in [ 1, \mathcal {N} _ {\rho} ],} \\ & {\frac {\partial \phi_ {C k}}{\partial t} + \zeta_ {C k} \phi_ {C k} = p, \quad \forall k \in [ 1, \mathcal {N} _ {C} ],} \end{array}\tag{7a}
$$

(7b) 

with zero initial conditions. Eqs. (6), coupled with Eqs. (7), form the augmented system of time-domain governing equations for wave propagation in porous materials. As indicated by Eqs. (4), the effects of frequency-dependent properties are now manifested by a superimposition of the frequency-independent asymptotic values $\{ \rho _ { m } , c _ { m } \}$ and the frequency-dependent auxiliary variables $\{ \phi _ { \rho k } , \phi _ { C k } \}$ . Compared to the relevant prior work on the ER boundary formulation [37], the proposed time-domain formulation completely excludes spatial derivatives of primary acoustic variables from the auxiliary differential Eqs. (7) thanks to application of the partial fractional decomposition in Eqs. (5) following [36,39]. Furthermore, since there is no spatial derivative of the auxiliary variables in the system, the part of the computational cost related to the spatial derivative approximation does not increase with the number of the auxiliary variables. As will be seen in the following Section 3.1, this formulation offers valuable convenience in the derivation of the spatial discretization scheme, and a consistent numerical treatment can be applied to discretize both the air and the porous material domains. It should be noted that the auxiliary variables need to be defined and time integrated in the same way as the primary acoustic variables in the material domain, resulting in an unavoidable increase in terms of the memory storage and the time integration cost. 

## 2.3. Interface conditions

The considered thin covering between the air (denoted by the subscript ??) and the porous material (denoted by ??) is modeled by an interface condition containing a pressure jump and continuous normal velocity [56], i.e., 

$$
\begin{array}{r} \mathbf {v} _ {a} \cdot \mathbf {n} _ {a} = - \mathbf {v} _ {m} \cdot \mathbf {n} _ {m}, \\ p _ {a} - p _ {m} = Z _ {t} \mathbf {v} _ {a} \cdot \mathbf {n} _ {a}, \end{array}\tag{8a}
$$

(8b) 

where ${ \mathbf { n } } _ { a }$ and $\mathbf { n } _ { m }$ are the unit outward normal vector at the boundary surface of each subdomain satisfying $\mathbf { n } _ { a } = - \mathbf { n } _ { m }$ . The transfer impedance of the thin covering is denoted as $Z _ { t }$ . It becomes zero when the porous material is in direct contact with the air domain. It is worth mentioning that the effect of covering thickness on sound absorption is taken into account by the transfer impedance model directly. Since the thickness of the covering is typically much smaller than the acoustic wavelength, the edge diffraction effect of the covering is neglected. This pressure jump interface model has been successfully applied to simulate sound transmission across permeable thin membranes [44–46], perforated plates [39] and microperforated panels [47 48] 

## 3. Numerical schemes

## 3.1. Spatial discretization with the DG method

To discretize the spatial derivative operators, we first rewrite the governing equations for the primary acoustic state variables in both subdomains (Eqs. (1) for the air and Eqs. (6) for the porous material), into a general first-order hyperbolic form as follows 

$$
\frac {\partial \mathbf {q}}{\partial t} + \mathbf {A} _ {x} \frac {\partial \mathbf {q}}{\partial x} + \mathbf {A} _ {y} \frac {\partial \mathbf {q}}{\partial y} + \mathbf {A} _ {z} \frac {\partial \mathbf {q}}{\partial z} + \mathbf {D q} = \mathbf {g},\tag{9}
$$

where $\mathbf { q } ( \mathbf { x } , t ) = [ v _ { x } , v _ { y } , v _ { z } , p ] ^ { \mathrm { T } }$ denotes the primary acoustic variable vector. The flux Jacobian matrices $\mathbf { A } _ { i } ~ ( j \in \{ x , y , z \} )$ 

$$
\mathbf {A} _ {j} = \left[ \begin{array}{c c c c} 0 & 0 & 0 & \frac {\delta_ {x j}}{\rho} \\ 0 & 0 & 0 & \frac {\delta_ {y j}}{\rho} \\ 0 & 0 & 0 & \frac {\delta_ {z j}}{\rho} \\ \rho c ^ {2} \delta_ {x j} & \rho c ^ {2} \delta_ {y j} & \rho c ^ {2} \delta_ {z j} & 0 \end{array} \right],
$$

(10) 

have frequency-independent constant entries, where the density ?? and speed of sound ?? takes the corresponding constant pair of values of each subdomain, i.e., $\{ \rho , c \} = \{ \rho _ { a } , c _ { a } \}$ for the air while $\{ \rho , c \} = \{ \rho _ { m } , c _ { m } \}$ for the porous material, and $\delta _ { i j }$ denotes the Kronecker delta function. The relaxation matrix ??, and the right-hand side source-like term ?? are fully responsible for the frequency-(in)dependency of the propagation medium. Specifically, for the air domain with frequency-independent medium properties, both ?? and ?? are null. For the porous material domain, the relaxation matrix ?? and term ?? are functions of the frequency-dependent auxiliary variables $\{ \phi _ { \rho k } , \phi _ { C k } \}$ as 

$$
\mathbf {D} = \left[ \begin{array}{c c c c} \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 & 0 & 0 \\ 0 & \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 & 0 \\ 0 & 0 & \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} & 0 \\ 0 & 0 & 0 & \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \end{array} \right], \quad \mathbf {g} = \left[ \begin{array}{c} \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {x} \\ \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {y} \\ \frac {1}{\rho_ {m}} \sum_ {k = 1} ^ {\mathcal {N} _ {\rho}} B _ {\rho k} \zeta_ {\rho k} \phi_ {\rho k} ^ {z} \\ \rho_ {m} c _ {m} ^ {2} \sum_ {k = 1} ^ {\mathcal {N} _ {C}} B _ {C k} \zeta_ {C k} \phi_ {C k} \end{array} \right].\tag{11}
$$

Since the frequency-dependent terms ?? and ?? are isolated from the flux Jacobian matrices, the air–material coupled system to be spatially discretized is transformed into one with piece-wise constant material properties. 

To solve Eq. (9) with the DG method, the physical domain ?? is partitioned into a set of non-overlapping elements $\varOmega ^ { e } .$ . Following the nodal DG formulation [57,58], in each element $\varOmega ^ { e }$ , a local piecewise polynomial approximation of the unknown solution $\mathbf { q } _ { h } ^ { e } ( { \bf x } , t )$ is expressed by: 

$$
\mathbf {q} ^ {e} (\mathbf {x}, t) \approx \mathbf {q} _ {h} ^ {e} (\mathbf {x}, t) = \sum_ {i = 1} ^ {N _ {p}} \mathbf {q} _ {h} ^ {e} (\mathbf {x} _ {i} ^ {e}, t) l _ {i} ^ {e} (\mathbf {x}),\tag{12}
$$

where the subscript ℎ denotes the numerical approximation, $\mathbf { q } _ { h } ^ { e } ( \mathbf { x } _ { i } ^ { e } , t ) = [ v _ { x h } ^ { e } , v _ { \nu h } ^ { e } , v _ { z h } ^ { e } , p _ { h } ^ { e } ] ^ { \mathrm { T } }$ are the unknown nodal values at locations $\mathbf { x } _ { i } ^ { e } , ~ l _ { i } ^ { e } ( \mathbf { x } )$ is the multi-dimensional Lagrange polynomial basis of order ?? satisfying $l _ { i } ^ { e } ( { \bf x } _ { i } ^ { e } ) = \delta _ { i j } . ~ N _ { p }$ is the number of local basis functions (the degree of freedom) inside a single element and is equal to $( N + d ) ! / ( N ! d ! )$ for simplex elements, where ?? is the dimensionality. The basis functions $l _ { i } ^ { e } ( { \bf x } )$ are determined by the nodal distribution ${ \bf x } _ { i } ^ { e } ,$ , and in this study, the Legendre–Gauss–Lobatto (LGL) quadrature points are used for 1D problems and the ??-optimized nodal distribution [57] are used for multi-dimensional elements. The same local polynomial approximations are applied to the auxiliary variables $\phi _ { \rho k }$ and $\phi _ { C k }$ in the porous material domain $\varOmega _ { m }$ 

The variational formulation is obtained by multiplying Eq. (9) with a local test function $l _ { i } ^ { e } ( { \bf x } )$ and integration by parts twice, yielding 

$$
\int_ {\Omega^ {e}} l _ {i} ^ {e} \left(\frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial t} + \mathbf {A} _ {x} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial x} + \mathbf {A} _ {y} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial y} + \mathbf {A} _ {z} \frac {\partial \mathbf {q} _ {h} ^ {e}}{\partial z} + \mathbf {D} \mathbf {q} _ {h} ^ {e} - \mathbf {g}\right) d \mathbf {x} = \oint_ {\partial \Omega^ {e}} l _ {i} ^ {e} \left(\mathbf {A} _ {n} ^ {e} \mathbf {q} _ {h} ^ {e} - \mathbf {F} ^ {e} (\mathbf {q} _ {h} ^ {e}, \mathbf {q} _ {h} ^ {e +})\right) d \mathbf {x},\tag{13}
$$

where $\mathbf { A } _ { n } ^ { e } : = \mathbf { A } _ { x } n _ { x } ^ { e } + \mathbf { A } _ { y } n _ { \nu } ^ { e } + \mathbf { A } _ { z } n _ { z } ^ { e }$ is the normal flux matrix along the outward normal vector $\mathbf { n } _ { e } = [ n _ { x } ^ { e } , n _ { v } ^ { e } , n _ { z } ^ { e } ]$ of the element interface ????<sup>??</sup>. ${ \bf F } ^ { e } ( { \bf q } _ { h } ^ { e } , { \bf q } _ { h } ^ { e + } )$ , the so-called numerical flux across element surface $\partial \varOmega ^ { e } ,$ , is a function of both the local solution value $\mathbf { q } _ { h } ^ { e }$ and the solution value ${ \bf q } _ { h } ^ { e + }$ of the neighboring element from the other side of the interface surface. As is known. the numerical flux is a paramount factor on the stability and accuracy of the DG method [59]. Apart from linking neighboring interior elements, the numerical flux also serves to impose boundary conditions weakly and to guarantee the stability of the formulation. Typically, three types of fluxes, central flux, Lax–Friedrich flux, and Riemann solver (i.e., upwind flux or Godunov flux), are used. The central flux is the simplest scheme that just takes the average values on both sides of an interface and is non-dissipative by nature. The Lax-Friedrich flux stabilizes the DG method by straightforwardly adding a dissipative term based on the maximum wave propagation speed. Both of them are vulnerable to a long-time instability issue due to the lack of proper dissipation to eliminate spurious modes [60,61]. By contrast, the upwind flux based on the Riemann solver has supreme dissipation and dispersion properties [62] by considering underlying physics and eliminating spurious modes properly. Therefore, it is used in this work. In the following section, the numerical flux formulation is derived for the element surface aligned with the jump interface between the air and the porous material, which can be straightforwardly applied to the simplified case of interior element interfaces and to the scenario where the thin covering is absent. 

## 3.2. Riemann solver across air–material interface

The upwind numerical flux is constructed based on the solutions of the Riemann problem, which considers the interface between two homogeneous media with frequency-independent properties as represented by the constant flux Jacobian matrices. Following the notation convention [57,63,64], the superscripts ‘‘ − ’’ and ‘‘ + ’’ are used to emphasize the values inside these two media. In the remainder, ?? $: = { \bf n } ^ { - } = [ n _ { x } , n _ { y } , n _ { z } ]$ denotes the outward unit normal vector. The constant media properties are denoted by $\{ \rho ^ { - } , c ^ { - } \}$ in the inward direction of ?? and $\{ \rho ^ { + } , c ^ { + } \}$ in the outward direction of ??. Mathematically, the Riemann problem is to solve Eqs. (9) for the given piecewise constant medium with the discontinuous initial condition 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/29d3e3a6522a5bbb83c8671c96539b0694d8979ede2eece59f643f95b5883fec.jpg)



Fig. 1. Illustration of the Rankine-Hugoniot jump conditions for the Riemann problem. Characteristics with distinct wave propagation speeds and directions are denoted by the arrow lines.


$$
\mathbf {q} _ {0} (\mathbf {x}) = \left\{ \begin{array}{l l} \mathbf {q} ^ {-} & \text {if} \quad \mathbf {n} \cdot (\mathbf {x} - \mathbf {x} _ {0}) <   0 \\ \mathbf {q} ^ {+} & \text {if} \quad \mathbf {n} \cdot (\mathbf {x} - \mathbf {x} _ {0}) > 0 \end{array} \right.
$$

where $\mathbf { x } _ { 0 }$ lies on the interface. To solve the Riemann problem, the characteristics of the system are first determined by the eigendecomposition of the flux matrix $\mathbf { A } _ { n }$ along ?? 

$$
\mathbf {A} _ {n} := \mathbf {A} _ {x} n _ {x} + \mathbf {A} _ {y} n _ {y} + \mathbf {A} _ {z} n _ {z} = \mathbf {R} \boldsymbol {\Lambda} \mathbf {R} ^ {- 1},\tag{14}
$$

where the eigenmatrix reads 

$$
\mathbf {R} = \frac {1}{2} \left[ \begin{array}{c c c c} - n _ {x} & - 2 n _ {y} & - 2 n _ {z} & n _ {x} \\ - n _ {y} & 2 n _ {x} & 0 & n _ {y} \\ - n _ {z} & 0 & 2 n _ {x} & n _ {z} \\ \rho c & 0 & 0 & \rho c \end{array} \right],\tag{15}
$$

and Λ is the eigenvalue matrix with diagonal entries given by $\lambda _ { 1 } = - c , \lambda _ { 2 } = \lambda _ { 3 } = 0 , \lambda _ { 4 } = c ,$ representing the characteristic wave speed. The Rankine–Hugoniot (RH) jump condition [57,63,64] 

$$
- \lambda_ {i} (\mathbf {q} ^ {\text { neg }} - \mathbf {q} ^ {\text { pos }}) + \mathbf {A} _ {n} (\mathbf {q} ^ {\text { neg }} - \mathbf {q} ^ {\text { pos }}) = \mathbf {0}\tag{16}
$$

holds across each characteristic wave of speed $\lambda _ { i } ( i \in \{ 1 , 2 , 3 , 4 \} )$ , where $\mathbf { q } ^ { \mathrm { n e g } }$ is the state vector in the negative normal direction across the characteristic wave and ${ \bf q } ^ { \mathrm { p o s } }$ is the state vector in the positive normal direction. Matrix $\mathbf { A } _ { n }$ is evaluated using the corresponding material property value $\{ \rho , c \}$ in the region where the $\lambda _ { i }$ characteristic wave travels as shown in Fig. 1, which is emphasized by the superscripts $^ { 6 6 } - { } ^ { 5 5 }$ and $^ { 6 \prime } + \boldsymbol { ^ { \prime \prime } }$ . The above RH jump condition (16) can be further exemplified as 

$$
\begin{array}{c} c ^ {-} (\mathbf {q} ^ {-} - \mathbf {q} ^ {a}) + \mathbf {A} _ {n} ^ {-} (\mathbf {q} ^ {-} - \mathbf {q} ^ {a}) = \mathbf {0}, \\ \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} - \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {0}, \\ - c ^ {+} (\mathbf {q} ^ {b} - \mathbf {q} ^ {+}) + \mathbf {A} _ {n} ^ {+} (\mathbf {q} ^ {b} - \mathbf {q} ^ {+}) = \mathbf {0}. \end{array}\tag{17a}
$$

(17b) 

(17c) 

Two unknown intermediate states $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ are the actual Riemann solutions for the considered system. Based on the mathematical definition of eigenvectors of $\mathbf { A } _ { n } ,$ the RH condition equivalently states that the jumps in state vector are a linear combination of that corresponding side’s eigenvectors [63,65]: 

$$
\begin{array}{r} \mathbf {q} ^ {-} - \mathbf {q} ^ {a} = \alpha_ {1} \mathbf {r} _ {1} ^ {-}, \\ \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} - \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {0}, \\ \mathbf {q} ^ {b} - \mathbf {q} ^ {+} = \alpha_ {4} \mathbf {r} _ {4} ^ {+}, \end{array}\tag{18a}
$$

(18b) 

(18c) 

where $\mathbf { r } _ { i }$ denotes the ??th column of $\mathbf { R } ,$ and ${ \pmb { \alpha } } = [ \alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } , \alpha _ { 4 } ]$ denotes the characteristic coefficients to be solved, corresponding to the characteristic wave of speed $\lambda _ { i } ~ ( i \in \{ 1 , 2 , 3 , 4 \} )$ ). Note that characteristic waves with zero speed do not contribute to the upwind numerical flux, therefore $\alpha _ { 2 }$ and $\alpha _ { 3 }$ are not considered. 

Mathematically speaking, the system of the RH jump condition (18) is underdetermined and needs extra conditions in order to be solved uniquely. From the physical perspective, connections between the two sides across the interface are supposed to be established by imposing appropriate physical boundary conditions over the intermediate states $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ . Solving Eqs. (18) with the interface condition of Eq. (8) yields the solutions of characteristic coefficient $\{ \alpha _ { 1 } , \alpha _ { 4 } \}$ as 

$$
\left[ \begin{array}{c} \alpha_ {1} \\ \alpha_ {4} \end{array} \right] = \frac {2}{Z ^ {-} + Z _ {t} + Z ^ {+}} \left[ \begin{array}{c} p ^ {-} - p ^ {+} - Z ^ {+} v _ {n} ^ {+} - (Z _ {t} + Z ^ {+}) v _ {n} ^ {-} \\ p ^ {-} - p ^ {+} + Z ^ {-} v _ {n} ^ {-} + (Z _ {t} + Z ^ {-}) v _ {n} ^ {+} \end{array} \right],\tag{19}
$$

where the characteristic impedance of the two homogeneous media $Z ^ { - } = \rho ^ { - } c ^ { - }$ and $Z ^ { + } = \rho ^ { + } c ^ { + }$ are introduced and $v _ { n } ^ { + } = \mathbf { v } ^ { + } \cdot \mathbf { n } ^ { + } =$ $- \mathbf { v } ^ { + } \cdot \mathbf { n } , v _ { n } ^ { - } = \mathbf { v } ^ { - } \cdot \mathbf { n }$ . Substitution of the calculated ?? from Eq. (19) into Eq. (18) yields the Riemann solutions $\{ \mathbf { q } ^ { a } , \mathbf { q } ^ { b } \}$ }. For the element in the inward direction of ??, the upwind numerical flux ?? along ?? reads 

$$
\mathbf {F} = \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {a} = \mathbf {A} _ {n} ^ {-} \mathbf {q} ^ {-} + c ^ {-} \alpha_ {1} \mathbf {r} _ {1} ^ {-},\tag{20}
$$

while, for the element on the other side of the interface, the upwind numerical flux ?? along ?? is 

$$
\mathbf {F} = \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {b} = \mathbf {A} _ {n} ^ {+} \mathbf {q} ^ {+} + c ^ {+} \alpha_ {4} \mathbf {r} _ {4} ^ {+}.\tag{21}
$$

Now return back to the physical setting of the acoustic wave propagation across the covered porous material interface. All quantities with superscripts $^ { 6 6 } - { } ^ { 5 5 }$ and $" + "$ are evaluated using the values of air and material domain, and denoted with subscripts ?? and $m ,$ respectively. For the elements along the air domain boundary $\varOmega _ { a } ,$ the upwind numerical flux $\mathbf { F } ^ { a }$ in the direction of $\boldsymbol { \mathbf { n } } _ { a } : = \boldsymbol { \mathbf { n } } = - \boldsymbol { \mathbf { n } } _ { m }$ follows (20) and can be written compactly as 

$$
\mathbf {F} ^ {a} = \mathbf {R} _ {a} \boldsymbol {\Lambda} _ {a} \left[ \begin{array}{c} \mathcal {R} _ {a m} \varpi_ {a} ^ {o} + \mathcal {T} _ {m a} \varpi_ {m} ^ {o} \\ 0 \\ 0 \\ \varpi_ {a} ^ {o} \end{array} \right],\tag{22}
$$

where 

$$
\begin{array}{r} \varpi_ {a} ^ {o} = \frac {p _ {a}}{Z _ {a}} + \mathbf {v} _ {a} \cdot \mathbf {n} _ {a}, \\ \varpi_ {m} ^ {o} = \frac {p _ {m}}{Z _ {m}} + \mathbf {v} _ {m} \cdot \mathbf {n} _ {m}, \end{array}\tag{23a}
$$

(23b) 

are the outgoing characteristic waves traveling towards the interface from the air and the porous material side, respectively. Here, two coefficients 

$$
\mathcal {R} _ {a m} = \frac {Z _ {t} + Z _ {m} - Z _ {a}}{Z _ {t} + Z _ {m} + Z _ {a}}, \quad \mathcal {T} _ {m a} = \frac {2 Z _ {m}}{Z _ {t} + Z _ {m} + Z _ {a}},\tag{24}
$$

are introduced and named as the reflection coefficients $\mathcal { R } _ { a m }$ from air to material domain and the transmission coefficients $\tau _ { m a }$ from material to air domain. The physical interpretations of these coefficients will be discussed further in Section 3.2.1. 

Similarly, for the element inside the material domain, the upwind numerical flux $\mathbf { F } ^ { m }$ along ${ \mathbf { n } } _ { m }$ is the negative of expression (21) due to the sign change of the normal vector, that is 

$$
\mathbf {F} ^ {m} = \mathbf {R} _ {m} \boldsymbol {\Lambda} _ {m} \left[ \begin{array}{c} \mathcal {R} _ {m a} \varpi_ {m} ^ {o} + \mathcal {T} _ {a m} \varpi_ {a} ^ {o} \\ 0 \\ 0 \\ \varpi_ {m} ^ {o} \end{array} \right],\tag{25}
$$

where 

$$
\mathcal {R} _ {m a} = \frac {Z _ {t} + Z _ {a} - Z _ {m}}{Z _ {t} + Z _ {m} + Z _ {a}}, \quad \mathcal {T} _ {a m} = \frac {2 Z _ {a}}{Z _ {t} + Z _ {m} + Z _ {a}}.\tag{26}
$$

## 3.2.1. Physical interpretations of <sup></sup> and <sup></sup> and unified numerical formulations

The concept of characteristics plays a pivotal role in numerically solving initial–boundary value problems of linear hyperbolic system since it fundamentally connects the algebraic structure of the system with the represented physical phenomena of wave propagation. It is known from the theoretical normal mode analysis [66–68] that the characteristic variety of the system admits solution expansions of modes in the form of plane waves of all wave numbers. Consequently, the incoming and outgoing characteristic wave components can be interpreted as plane waves from a physical point of view [69]. For the same reason, the coefficients <sup></sup> and <sup></sup> introduced in Eqs. (24) and (26) match exactly the corresponding plane wave reflection and transmission coefficients of the particle velocity of plane waves traveling across an impedance discontinuity interface [70]. 

When the wave propagates inside a homogeneous medium, it experiences no reflection. Then, <sup></sup> reduces to zero and <sup></sup> remains unity. For example, for a element $\varOmega ^ { e }$ inside the air subdomain, the upwind flux ${ \bf F } ^ { e } ( { \bf q } _ { h } ^ { e } , { \bf q } _ { h } ^ { e + } )$ at the interior element surface $\partial \mathcal { Q } ^ { e }$ becomes: 

$$
\mathbf {F} ^ {e} (\mathbf {q} _ {h} ^ {e}, \mathbf {q} _ {h} ^ {e +}) = \mathbf {R} _ {a} \boldsymbol {\Lambda} _ {a} \left[ \begin{array}{c} \frac {p _ {h} ^ {e +}}{Z _ {a}} - \mathbf {v} _ {h} ^ {e +} \cdot \mathbf {n} _ {e} \\ 0 \\ 0 \\ \frac {p _ {h} ^ {e}}{Z _ {a}} + \mathbf {v} _ {h} ^ {e} \cdot \mathbf {n} _ {e} \end{array} \right].\tag{27}
$$

Therefore, a unified numerical treatment of exact upwind flux can be realized for both interior elements and boundary elements. 

## 3.2.2. Incorporation of frequency-dependency

So far, the presented formulations are limited to the covering interface with a frequency-independent real-valued transfer impedance $Z _ { t } .$ . However, covering materials typically exhibit frequency-dependent acoustic properties and models of various levels of complexity exist for $Z _ { t } , e . g .$ ., the classical Maa’s model for microperforated panel [71]. In order to incorporate general broadband models, the same numerical strategy from previous work [69] is adopted. First, similar to the numerical treatments to $\rho _ { \mathrm { e f } } ( \omega )$ and $c _ { \mathrm { e f } } ( \omega )$ in Eqs. (4), the plane wave coefficients $\mathcal { R } _ { m a } ( \omega ) , \mathcal { R } _ { a m } ( \omega ) , \mathcal { T } _ { m a } ( \omega )$ and $\tau _ { a m } ( \omega )$ are expressed in the form of the multi-pole model in the frequency-domain. Then, these coefficients are convolved in time with the corresponding outgoing characteristic waves as defined in Eqs. (22) and (25) to obtain the incoming waves, during which the ADE method is used to calculate the convolutions. 

In this work, a limp permeable membrane model is considered as a representative example. Its acoustic characteristics are mainly governed by two material parameters, namely the surface mass density denoted by ?? [kg $\mathrm { m } ^ { - 2 } ]$ and the flow resistance $r _ { f } \ [ \mathrm { P a } \ s \ \mathrm { m } ^ { - 1 } ]$ representing the effects of sound induced vibration and air permeability respectively. Analogous to the electric impedance of two circuit elements in parallel, the transfer impedance reads [56] 

$$
Z _ {t} (\omega) = \left(\frac {1}{r _ {f}} + \frac {1}{\mathrm{i} \omega m}\right) ^ {- 1}.\tag{28}
$$

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

with $B _ { t } = 2 / ( r _ { f } + Z _ { a } + Z _ { m } )$ and $\zeta _ { t } = r _ { f } ( Z _ { a } + Z _ { m } ) / ( m Z _ { a } + m Z _ { m } + m r _ { f } )$ 

## 3.3. Semi-discrete formulation

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

where vectors $\phi _ { \rho k } ^ { x e } , \phi _ { \rho k } ^ { y e } , \phi _ { \rho k } ^ { z e }$ , and $\phi _ { C k } ^ { e }$ denote the local unknown nodal value vectors of auxiliary variables that are coupled with primary nodal values by Eq. (7). 

## 3.4. Temporal discretization

After the spatial discretization by the DG method, the total semi-discrete system can be expressed in a general form of ODEs as: 

$$
\frac {\partial \tilde {\mathbf {q}} _ {h}}{\partial t} = \mathcal {L} \big (\tilde {\mathbf {q}} _ {h} (t), t \big),\tag{33}
$$

where ${ \tilde { \mathbf { q } } } _ { h }$ is the unknown vector including primary acoustic variables ${ \bf q } _ { h }$ and auxiliary variables $\{ \phi _ { \rho k } , \phi _ { C k } \}$ from Eqs. (7) . Here, <sup></sup> is the operator that considers both the DG spatial discretization of Eq. (13) and auxiliary differential Eqs. (7). This system is temporally integrated using the Taylor Series time integrator (TSI) scheme [72], resulting in a final time-discrete formulation as 

$$
\tilde {\mathbf {q}} _ {h} (t + \Delta t) = \tilde {\mathbf {q}} _ {h} (t) + \sum_ {i = 1} ^ {N _ {t}} \frac {\Delta t ^ {i}}{i !} \mathcal {L} ^ {i} \tilde {\mathbf {q}} _ {h} (t),\tag{34}
$$

where $N _ { t }$ denotes the temporal order of accuracy and $\varDelta t = t ^ { n + 1 } - t ^ { n }$ is the time step. 

As an explicit time-stepping method, the TSI scheme is subject to the Courant–Friedrichs–Lewy (CFL) conditional stability that sets an upper bound on the time step size. For a discretization combination of ??th order in space with the DG method and a (?? +1)th order in time with the TSI scheme, the time step size follows [73] 

$$
\Delta t = C _ {C F L} \Delta x _ {l} \frac {1}{c} \frac {1}{(2 N + 1)},\tag{35}
$$

where $C _ { C F L }$ is a constant of order <sup></sup>(1), ?? is the constant wave speed of the medium, and $\varDelta x _ { l }$ is a measure of the element size. For the porous material domain, apart from the DG spatial operator, the time step size is additionally bounded by the stiffness of auxiliary differential equations, $i e .$ , the fitting parameters $\{ \zeta _ { \rho } , \zeta _ { C } \}$ as in Eqs. (4). It was found [74] that when vector fitting is used, the maximum value of $\zeta _ { \rho }$ and $\zeta _ { C }$ increases with the number of poles. To improve the computational efficiency, an explicit local time-stepping strategy accompanying the TSI scheme [72] is employed. Consequently, a smaller time step size can be used in the porous material subdomain to ensure that the product ?? ⋅ ???? falls into the stability region of the time-integration scheme. The TSI scheme, similar to the multi-stage Runge–Kutta scheme [75], has a larger stability region with increasing order of accuracy $N _ { t }$ . For example, a fourth-order TSI scheme admits a maximum value of $\zeta \cdot \varDelta t = 2 . 7 7 5$ . Further discussion on numerical stability and details of implementations can be found in the Ref. [72]. 

## 4. Numerical results

## 4.1. Poles identification of porous materials

In this work, two kinds of rigid-frame porous material characterized by JCAL model are considered: melamine foam and glass wool. These materials are used in earlier studies [7,53]. Table 1 shows their model parameters. As a reference, their theoretical normal incidence sound absorption coefficient are shown in Fig. 2. Their effective density $\rho _ { \mathrm { e f } }$ and effective compressibility $c _ { \mathrm { e f } }$ as a function of angular frequency ?? are described as [29] 

$$
\begin{array}{r l} & {\rho_ {\mathrm{ef}} (\omega) = \frac {\rho_ {a} \alpha_ {\infty}}{\varphi} \Big [ 1 + \frac {\sigma \varphi}{\mathrm{i} \omega \alpha_ {\infty} \rho_ {a}} \big (1 + \frac {4 \mathrm{i} \alpha_ {\infty} ^ {2} \eta \rho_ {a}}{\sigma^ {2} \Lambda^ {2} \varphi^ {2}} \big) ^ {1 / 2} \Big ],} \\ & {\mathcal {C} _ {\mathrm{ef}} (\omega) = \frac {\varphi}{\rho_ {a} c _ {a} ^ {2}} \Bigg (\gamma - \frac {\gamma - 1}{\big [ 1 + \frac {\varphi \eta}{\mathrm{i} \omega k _ {0} ^ {\prime} \rho_ {a} P _ {r}} (1 + \frac {4 \mathrm{i} \omega k _ {0} ^ {\prime 2} \rho_ {a} P _ {r}}{\eta \Lambda^ {\prime 2} \varphi^ {2}}) ^ {1 / 2} \big ]} \Bigg).} \end{array}\tag{36a}
$$

(36b) 

The limiting constant values in Eqs. (4) can be calculated analytically as $\rho _ { m } = \rho _ { a } \alpha _ { \infty } / \varphi$ and $C _ { m } = \varphi / ( \rho _ { a } c _ { a } ^ { 2 } )$ . The frequency vector needed as input for the VF algorithm is set to [20 ∶ 2000] Hz with a resolution of 1 Hz. As expected for dissipative porous materials, the VF algorithm selects only real poles automatically. Same as findings from Ref. [53], numerical tests show that the relative root squared error diminishes fast with increasing number of poles $\{ N _ { \rho } , N _ { C } \}$ , and drops below $1 0 ^ { - 5 }$ when $\{ N _ { \rho } , N _ { C } \} > 5$ 


Table 1



Properties of porous materials


<table><tr><td>Property</td><td>Glass wool</td><td>Melamine foam</td></tr><tr><td>Flow resistivity σ [N s m-4]</td><td>70821</td><td>4500</td></tr><tr><td>Porosity φ</td><td>0.967</td><td>0.99</td></tr><tr><td>Tortuosity α∞</td><td>1.049</td><td>1.0</td></tr><tr><td>Viscous length Λ [m]</td><td>6 × 10-5</td><td>1.3 × 10-4</td></tr><tr><td>Thermal length Λ&#x27; [m]</td><td>1.4 × 10-4</td><td>1.6 × 10-4</td></tr><tr><td>Thermal permeability k0&#x27; [m2]</td><td>6.345 × 10-9</td><td>4.0 × 10-9</td></tr><tr><td>Dynamic viscosity η [N m-2]</td><td>1.82 × 10-5</td><td>1.82 × 10-5</td></tr><tr><td>Prandtl number Pr</td><td>0.71</td><td>0.71</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/77b6caf1c45e9c315e7636c70a254a6b5c3d64f6b4938ef31550a568b099214f.jpg)



Fig. 2. Theoretical normal incidence absorption coefficient of simulated porous materials.


## 4.2. 1D tests: verification against analytical solutions

To validate the proposed numerical scheme, 1D numerical tests are performed. The computational domain includes an air subdomain $x \in [ - 1 . 5 , 0 ]$ m and a porous material $x \in [ 0 , 1 . 5 ]$ m subdomain made of melamine foam, which are separated by a pressure jump interface at $x = 0 .$ . Totally sound absorbing boundary conditions are applied at both ends of the domain. Without loss of generality, the interface is acoustically characterized as the limp permeable membrane introduced in Section 3.2.2 and has a surface mass densit $m = 0 . 1 3$ kg $\mathrm { m } ^ { - 2 }$ and a flow resistance $r _ { f } = 6 4 ~ \mathrm { N } ~ s ~ \mathrm { m } ^ { - 3 }$ , which is representative of general purpose fabrics. This numerical test is inspired by the work of [39], where an analytical solution in time is provided in detail. The simulations in the air subdomain are initiated by a Mexican hat pulse: 

$$
\begin{array}{r l} & p (x, t = 0) = \big (1 - \big [ \frac {x - x _ {s}}{B} \big ] ^ {2} \big) \mathrm{e} ^ {- (\frac {x - x _ {s}}{\sqrt {2} B}) ^ {2}}, \\ & u (x, t = 0) = \frac {1}{\rho_ {a} c _ {a}} \big (1 - \big [ \frac {x - x _ {s}}{B} \big ] ^ {2} \big) \mathrm{e} ^ {- (\frac {x - x _ {s}}{\sqrt {2} B}) ^ {2}}, \end{array}\tag{37a}
$$

(37b) 

which is centered around $x _ { s } = - 1$ m and has sufficient energy up to 4000 Hz with width $B = 0 . 0 4 5$ m. In the material subdomain, both acoustic and auxiliary variables are initialized to zero. 

The element size is set as $\varDelta x = 0 . 1$ m and 5th order discretization schemes are used for both space and time, i.e., $N = N _ { t } = 5 .$ $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ of the melamine foam are approximated with 6 real poles, resulting in a root squared relative error of magnitude $\mathcal { O } ( 1 0 ^ { - 5 } )$ up to 4 kHz. Fig. 3(a) depicts the pressure fields at three time instants. The pulse at the time $\bar { t } = 1 . 5$ ms is the incident wave traveling towards the interface. Then, at $\bar { t } = 2 . 9$ ms, the pulse interacts with the interface, across which a pressure jump is well captured by the numerical scheme. The reflected pulse and the transmitted pulse can be observed at $\bar { t } = 3 . 6$ ms. Fig. 3(b) shows the absolute error of a consistent order of magnitude with respect to the analytical solutions. 

Aiming for a rigorous error analysis, the error of the numerical solution is split into two parts from two sources, namely the model error induced by the truncation of multipole model approximation of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ from Eq. (4), and the numerical error from the spatial and temporal discretization. To be specific, suppose $p _ { \mathrm { n u m } }$ denotes the numerical solution, ${ p } _ { \mathrm { a n a } }$ is the analytical solution calculated based on the exact values of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } } ,$ and $P _ { \mathrm { a n a * } }$ is the analytical solution built from truncated multipole model expressions of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ . These two types of relative error at a time ??<sup>̄</sup> are measured globally by [39]: 

$$
\epsilon_ {\mathrm{num}} (\bar {t}) = \frac {\| p _ {\mathrm{ana*}} (\bar {t}) - p _ {\mathrm{num}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana*}} (\bar {t}) \| _ {L ^ {2}}}\tag{38}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/efe34afcf5532b448fc874208cafc29467f9a5c54bfcc554652d548cd6542686.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/a4a5f9568f306a880462f2d6b609d1d7673934192eac686a2d0682308a7bd1e7.jpg)



(b)



Fig. 3. (a) Simulated pressure field in the air and the melamine foam with fabrics covering at three time instants; (b) error of the pressure field.


$$
\epsilon_ {\mathrm{model}} (\bar {t}) = \frac {\| p _ {\mathrm{ana*}} (\bar {t}) - p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}},\tag{39}
$$

where $\begin{array} { r } { \| f \| _ { L ^ { 2 } } = ( \int _ { \Omega } f ^ { 2 } ( \mathbf x ) \mathrm { d } \mathbf x ) ^ { 1 / 2 } } \end{array}$ denotes the $L ^ { 2 }$ norm and is calculated with the polynomial basis up to the order of approximation. The total relative error is measured globally by 

$$
\epsilon_ {\mathrm{tot}} (\bar {t}) = \frac {\| p _ {\mathrm{ana}} (\bar {t}) - p _ {\mathrm{num}} (\bar {t}) \| _ {L ^ {2}}}{\| p _ {\mathrm{ana}} (\bar {t}) \| _ {L ^ {2}}}\tag{40}
$$

To verify the global convergence rate, the domain is discretized with uniform mesh of various sizes. The time step $\varDelta t _ { a }$ in air subdomain varies proportionally to the mesh size ???? following the CFL condition (35) with $C _ { C F L }$ kept as a constant of 0.5. To accommodate the high stiffness of pole parameters $\{ \zeta _ { \rho } , \zeta _ { C } \}$ , the local time-stepping scheme [72] is used and the time step size in the material subdomain is set as $\Delta t _ { m } = 1 / 4 \Delta t _ { a }$ to ensure that the time integration is stable. As an example of demonstration, the convergence plot of using 5th order discretization schemes for both space and time, i.e., $N = N _ { t } = 5$ are shown in Fig. 4, where the global error at the time $\bar { t } = 4 . 4$ ms for both (a) the reflected wave in the air, and (b) the transmitted wave in the porous material are depicted. For larger mesh sizes and time steps, the total error is dominated by the numerical error, whereas for smaller mesh sizes, the total error converges to the modeling error. As expected, the numerical error is independent of the number of poles. As shown in Fig. 4, the convergence rate of numerical error matches the so-called ‘‘optimal’’ rate of convergence $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ for the DG method. 

For further assessments of the scheme, the local numerical accuracy around the pressure jump interface is quantified with the following error measure: 

$$
\epsilon_ {\mathrm{num}} ^ {l} (\bar {t}) = \frac {\left(\int_ {0} ^ {\bar {t}} | \lceil p \rceil_ {\mathrm{num}} - \lceil p \rceil_ {\mathrm{ana*}} | ^ {2} \mathrm{d} t\right) ^ {1 / 2}}{\left(\int_ {0} ^ {\bar {t}} \lceil p \rceil_ {\mathrm{ana*}} ^ {2} \mathrm{d} t\right) ^ {1 / 2}},\tag{41}
$$

where $\lceil p \rceil ( t )$ represents the time signal of the pressure jump across the interface of membrane, and the integral in time is calculated with the trapezoid rule. Simulations with an increasing number of order of accuracy are performed and the number of poles is fixed at 5. Fig. 5 depicts local numerical error $\epsilon _ { \mathrm { { n u m } } } ^ { l } ( \bar { t } )$ at $\bar { t } = 4 . 4$ ms. As we can see, in the asymptotic range of accuracy, the optimal rate $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ ) is achieved, whereas the error from coarser meshes and correspondingly larger time steps gradually follow the convergence rate of the trapezoid quadrature rule. Therefore, the proposed numerical treatment of the interface maintains the order of accuracy of the scheme. The same conclusions can be drawn from numerical tests with other material property values. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/2aac02e1de25a618b97144787e390ead1e0c5424968ad2b9553244a90bf4af1c.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/dd042b9477c9bbefdbeb06c111e65102809a41abf685c076dfc1b4fea2f13d32.jpg)



(b)



Fig. 4. $\epsilon _ { \mathrm { { t o t } } } , \epsilon _ { \mathrm { { n u m } } } ,$ and $\epsilon _ { m o d e l }$ for semi-infinite air and porous material with increasing number of real poles of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } } .$ . The limp permeable interface begets (a) a reflected wave in the air, and (b) a transmitted wave in the porous material.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/5a902df616ae1c1e7d0a57acc262ecfeb134ca8667b275a4469f18820b7b78bf.jpg)



Convergence behavior of local error around pressure jump interface.


## 4.3. Spherical wave reflection above rigidly-backed porous absorber

Now consider a single spherical wave reflection test case, where a monopole source is placed above a rigidly backed layer of porous absorber. This basic and fundamental case aims to assess the accuracy of the proposed scheme in 3D scenarios with other sources of modeling error avoided, $_ { e . g . }$ , geometrical error. Furthermore, an analytical sound pressure field solved by Allard et al. [76] is available for the case of a free interface between the porous material and the air, and is thus taken as an exact reference solution. As illustrated in Fig. 6, the free interface spans the horizontal ??–?? plane at $z = 0$ . A source is placed at $\mathbf { x _ { s } } = [ 0 , 0 , 0 . 3 5 ]$ m. Two receivers are placed at $\mathbf { x } _ { r 1 } = [ 0 , 0 , 0 . 0 2 ]$ m and $\mathbf { x } _ { r 2 } = [ 1 . 5 , 1 . 5 , 0 . 0 2 ]$ m, which corresponds to cases of a normal incidence and an oblique incidence with a specular reflection angle of $8 0 ^ { \circ }$ respectively. A Gaussian pressure pulse is used to initiate the simulations 

$$
p (\mathbf {x}, t = 0) = \mathrm{e} ^ {\frac {- \ln 2}{b ^ {2}} (\mathbf {x} - \mathbf {x} _ {s}) ^ {2}},\tag{42a}
$$

$$
\mathbf {v} (\mathbf {x}, t = 0) = \mathbf {0},\tag{42b}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/2923e223c00c7d75c7238a955e8719375cbce39086f387ecef6d7c51d2009a8f.jpg)



Fig. 6. Illustration of the numerical setup for the 3D test


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/28ba7192d4c72db2cc80e564a357f081b8a77453b2e116fe7cbb5fb6c4e7b234.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/5d6dd3e1e6d727ae0207482e788b0ae07bf0a37e15d106761dac1e07b8eb69de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0d0dddcf450d1e6fc02f434169477afcde0eede08e85597df95dbbcf5f46071c.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/be5a701817745e9df320538fb93bd21f3329d55da075c7cdf27ddbdf3412de75.jpg)



(b)



Fig. 7. Simulated and analytical pressures field above melamine foam at: (a) $\mathbf { x } _ { r 1 } , \theta ^ { \circ } = 0 ^ { \circ }$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$


with a half-bandwidth value of $b = 0 . 1 \textrm { m }$ , indicating a source spectrum up to 2 kHz. The overall computational domain has a dimension of $[ - 3 , 4 ] \times [ - 3 , 4 ] \times [ - d , 3 . 5 ]$ m, where ?? is the thickness of the porous material. Unstructured geometry conforming tetrahedra meshes are generated with a meshing software <sup>GMSH</sup> [77] for both the air and the porous material. Here, the mesh sizes are constrained by the thickness of the porous material, which is equal to the length of one side of the tetrahedron elements. In order to have sufficient spatial resolution, 7th order polynomial basis functions $( N = 7 )$ are used for the spatial discretization, resulting in roughly 9 volume-averaged degrees of freedom per wavelength at 2 kHz. The temporal order is set at $N _ { t } = 5$ to achieve a decent balance between the simulation accuracy and efficiency. The minimum radius of the inscribed sphere min $( r _ { i n } )$ is used as a measure of the element size $\varDelta x _ { l }$ . Among the performed numerical tests, the element size is the dominant restriction factor on the time step size. Hard wall boundary conditions are imposed on exterior boundaries of the whole computational domain, and the recorded pressure signals at receivers’ locations are cut before spurious reflections from exterior boundaries arrive. 

Fig. 7 shows the comparison of the simulated and analytical pressure spectra $\hat { p }$ in terms of the amplitude | ̂??| and the phase $\vartheta ( \hat { p } )$ for the case of melamine foam with two different thicknesses. The analytical solutions are calculated by using the exact values of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ as inputs whereas the numerical solutions are built on approximations of $\rho _ { \mathrm { e f } }$ and $c _ { \mathrm { e f } }$ with 5 real poles. The analytical free-field time solution of the initial Gaussian pulse is used in order to normalize its non-flat source power spectrum. The magnitude of simulated results have been normalized such that the free-field solution is of the form $\mathrm { e } ^ { - \mathrm { i } k r } / ( 4 \pi r )$ . Similarly, Fig. 8 displays the comparison results when the porous absorber is represented by the glass wool as shown in Table 1. It can be seen from these results that a good agreement between simulated and reference solutions is achieved, demonstrating the applicability of the proposed boundary scheme in 3D space and its high precision in a wide frequency range. 

Room acoustics features multiple reflections happen inside an enclosure. Additional numerical error is accumulated after each reflection. In the interest of a decent balance between the computational efficiency and accuracy, it is important to quantify the error in terms of the degrees of the freedom, which is proportional to the computational cost. For a certain broadband incident acoustic wave, the loss of sound pressure level and the distortion of the phase can be evaluated by the following dissipation error measure and $\epsilon _ { a m p }$ [dB] phase error measure $\epsilon _ { \vartheta }$ [%]: 

$$
\epsilon_ {\mathrm{amp}} (f) = 2 0 \log_ {1 0} \left| \frac {\hat {p} _ {\mathrm{num}} (f)}{\hat {p} _ {\mathrm{ana}} (f)} \right|,\tag{43a}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/f119484f68d86af3d00fd29c135eb83ce81257ac3c7ebdd92f1f865920259200.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/69e88d687eee10175ce7416ad49e56ee9980d9239eceec444d4df515f1f783d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0680e79aba9d688dca2ae324c8dc6f010da8a23ae4d942d6dc8a59c9c660915c.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/bb626ceace6fe15218e1d69747cbf0f3395cb08b748c23a32c87735acf0096b1.jpg)



(b)



Fig. 8. Simulated and analytical pressures field above glass wool at: (a) $\mathbf { x } _ { r 1 } , \theta ^ { \circ } = 0 ^ { \circ }$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/d2e9021351f062a4bfa038b9265e0359ae21ceaed40603481deacb29fb424f79.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/66ef5c2060a1ef0236d3d86b8632acb2cb9d6b98b9a6226b6b862ec7164b08d9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/0bf916740db2f59de5341a1620dbae12f919fe0d4ba019d3b98a04e0e242ee6d.jpg)



(a)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-07-14/fe8b1b0b-3ce1-4c94-a5b1-f11498298901/fd94b83c6ce20012c6178cea7b6ecca1f5da4a54f48bc714ad195f6680eec654.jpg)



(b)



Fig. 9. Dissipation error $\epsilon _ { a m p }$ [dB] and phase error $\epsilon _ { \vartheta }$ [%] of simulations at: (a) ?? $\theta ^ { \circ } = 0 ^ { \circ } ,$ , (b) $\mathbf { x } _ { r 2 } , \theta ^ { \circ } = 8 0 ^ { \circ }$


$$
\epsilon_ {\vartheta} (f) = \frac {1}{\pi} \big (\vartheta \big (\hat {p} _ {\mathrm{num}} (f) \big) - \vartheta \big (\hat {p} _ {\mathrm{ana}} (f) \big) \big) \times 100 \%.\tag{43b}
$$

Fig. 9 displays the dissipation error ?? and phase error ?? of the simulation results for both porous absorbers, which have markedly $\epsilon _ { a m p }$ $\epsilon _ { \vartheta }$ different values of flow resistivity $\sigma .$ . The combination of the mesh and the polynomial order results in spatial resolutions of 9 degrees of freedom per wavelength at 2. kHz. As expected. the numerical error maintains at a acceptable level when the spatial resolution is sufficient. It should be noted that the error in the low frequency range (below 500 Hz) is larger than the middle frequency range (500 − 1500 Hz). This is due to the premature cut of the recorded pressure signal, which leads to a loss of low-frequency power of the reflected sound field. Since the second receiver $\mathbf { x } _ { r 2 }$ of the oblique incidence case is closer to the outer hard wall boundary than the first receiver, the premature cut happens sooner. Consequently, the error in Fig. 9(b) is larger the one in Fig. 9(a). 

## 5. Conclusions

In this work, a numerical framework for modeling general extended reacting boundaries of porous materials including thin coverings has been developed for the purpose of time-domain room acoustic simulations. The equivalent fluid models are used to describe the acoustic wave propagation inside arbitrary porous materials, the effective density and compressibility of which are well approximated using multi-pole rational functions. By applying the ADE method to calculate the convolution integral, the resulting time-domain governing equations of porous materials can be written in a unified hyperbolic form like the linear acoustic equations for lossless air. Based on solving the underlying Riemann problem, a consistent upwind numerical flux formulation that ensures appropriate physical coupling between propagation media, including air, covering materials, and porous absorbers, is developed. The limp permeable membrane model is used to characterize the acoustic properties of the covering materials. To tackle potentia inefficiency issues due to the constrained time step size, the local time-stepping scheme is employed 

One-dimensional numerical tests verify the convergence property of the proposed formulation, where the optimal rate of convergence $\mathcal { O } ( \varDelta \boldsymbol { x } ^ { N + 1 } )$ for the DG method is obtained. Meanwhile, it is demonstrated that the interface coupling does not incur extra error. Three-dimensional tests further validate the capacity of the proposed methodology for representing practical extended reacting impedance boundaries in the multi-dimensional case. Both the magnitude and the phase information are accurately captured. 

The proposed numerical framework not only improves the applicability of the time-domain discontinuous Galerkin method for modeling room acoustics, but also can be applied to other wave-based methods, such as the finite volume method. Future work will investigate extension of the current framework to describe additional types of boundary materials, such as metamaterials. The implementation of more complicated surface covering models is an additional potential study topic. By employing the proposed model to provide reference solutions to complex room acoustic problems, it is possible to explore and test surrogate models of extended reacting boundaries that are more computationally efficient. 

## CRediT authorship contribution statement

Huiqing Wang: Conceptualization, Methodology, Software, Validation, Formal analysis, Writing – original draft, Writing – review & editing. Maarten Hornikx: Supervision, Writing – review & editing, Funding acquisition. 

## Data availability

No data was used for the research described in the article. 

## Acknowledgments

This work is funded by the Dutch Research Council (NWO) under grant No. 19430. 

## References



[11 J. Allard, N. Atalla, Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials, John Wiley & Sons, 2009. 





[2] L. Savioja, U.P. Svensson, Overview of geometrical room acoustic modeling techniques, J. Acoust. Soc. Am. 138 (2) (2015) 708–730. 





[3] T. Sakuma, S. Sakamoto, T. Otsuru, Computational Simulation in Architectural and Environmental Acoustics, Springer, 2014. 





[4] E. Brandão, A. Lenzi, S. Paul, A review of the in situ impedance and sound absorption measurement techniques, Acta Acust. United Acust. 101 (3) (2015) 443-463 





[5] A. Southern, S. Siltanen, D.T. Murphy, L. Savioja, Room impulse response synthesis and validation using a hybrid acoustic model, IEEE Trans. Audio, Speech, Lang. Process. 21 (9) (2013) 1940–1952. 





[6] S. Bilbao, B. Hamilton, J. Botts, L. Savioja, Finite volume time domain room acoustics simulation under general impedance boundary conditions, IEEE/ACM Trans. Audio. Speech Lang, Process. (TASLP) 24 (1) (2016) 161–173. 





[7] H. Wang, M. Hornikx, Time-domain impedance boundary condition modeling with the discontinuous Galerkin method for room acoustics simulations, J. Acoust, Soc, Am. 147 (4) (2020) 2534–2546. 





[8] E. Pind. A.P. Engsig-Karup. C.-H. Jeong, LS. Hesthaven, M.S. Meiling. J. Strømann-Andersen, Time domain room acoustic simulations using the spectra element method, J. Acoust. Soc. Am. 145 (6) (2019) 3299–3310. 





[9] T. Okuzono, T. Yoshida, K. Sakagami, Efficiency of room acoustic simulations with time-domain FEM including frequency-dependent absorbing boundary conditions: Comparison with frequency-domain FEM, Appl. Acoust. 182 (2021) 108212. 





[10] D.K. Wilson, S.L. Collier, V.E. Ostashev, D.F. Aldridge, N.P. Symons, D.H. Marlin, Time-domain modeling of the acoustic impedance of porous surfaces Acta Acust, United Acust, 92 (6) (2006) 965–975. 





[11] V.E. Ostashev, S.L. Collier, D.K. Wilson, D.F. Aldridge, N.P. Symons, D. Marlin, Padé approximation in time-domain boundary conditions of porous surfaces, J. Acoust. Soc. Am. 122 (1) (2007) 107–112. 





[12] P.M. Morse, K.U. Ingard, Theoretical Acoustics, Princeton University Press, 1986. 





[13] C.-H. Jeong, Guideline for adopting the local reaction assumption for porous absorbers in terms of random incidence absorption coefficients, Acta Acust. United Acust, 97 (5) (2011) 779–790. 





[14] R. Dragonetti, R.A. Romano, Considerations on the sound absorption of non locally reacting porous layers, Appl. Acoust. 87 (2015) 46–56. 





[15] R. Dragonetti, R.A. Romano, Errors when assuming locally reacting boundary condition in the estimation of the surface acoustic impedance, Appl. Acoust. 115 (2017) 121–130. 





[16] Y. Takahashi. T. Otsuru, R. Tomiku, In situ measurements of surface impedance and absorption coefficients of porous materials using two microphones and ambient noise, Appl. Acoust. 66 (7) (2005) 845–865. 





[17] R. Tomiku, T. Otsuru, N. Okamoto, T. Okuzono, T. Shibata, Finite element sound field analysis in a reverberation room using ensemble averaged surface normal impedance, in: INTER-NOISE and NOISE-CON Congress and Conference Proceedings, Vol. 2011, Institute of Noise Control Engineering, 2011, pp. 1780-1785 





[18] M. Aretz, M. Vorländer, Efficient modelling of absorbing boundaries in room acoustic FE simulations, Acta Acust. United Acust. 96 (6) (2010) 1042–1050. 





[19] Y. Yasuda, S. Ueno, M. Kadota, H. Sekine, Applicability of locally reacting boundary conditions to porous material layer backed by rigid wall: Wave-bas numerical study in non-diffuse sound field with unevenly distributed sound absorbing surfaces, Appl. Acoust. 113 (2016) 45–57. 





[20] M. Hodgson, A. Wareing, Comparisons of predicted steady-state levels in rooms with extended-and local-reaction bounding surfaces, J. Sound Vib. 30 (1–2) (2008) 167–177. 





[21] B. Yousefzadeh, M. Hodgson, Energy-and wave-based beam-tracing prediction of room-acoustical parameters using different boundary conditions, J. Acoust. Soc. Am. 132 (3) (2012) 1450–1461. 





[22] K. Gunnarsdóttir, C.-H. Jeong, G. Marbjerg, Acoustic behavior of porous ceiling absorbers based on local and extended reaction, J. Acoust. Soc. Am. 137 (1) (2015) 509–512. 





[23] M.A. Biot, Mechanics of deformation and acoustic propagation in porous media, J. Appl. Phys. 33 (4) (1962) 1482–1498. 





[24] M.A. Biot, Generalized theory of acoustic propagation in porous dissipative media, J. Acoust. Soc. Am. 34 (9) (1962) 1254–1264. 





[25] E. Deckers, N.-E. Hörlin, D. Vandepitte, W. Desmet, A Wave Based Method for the efficient solution of the 2D poroelastic Biot equations, Comput. Methods Appl. Mech. Engrg. 201 (2012) 245–262. 





[26] J.-D. Chazot, E. Perrey-Debain, B. Nennig, The partition of unity finite element method for the simulation of waves in air and poroelastic media, J. Acoust. Soc. Am. 135 (2) (2014) 724–733. 





[27] Y. Miki, Acoustical properties of porous materials-modifications of Delany-Bazley models, J. Acoust. Soc. Japan (E) 11 (1) (1990) 19–24. 





[28] J.-F. Allard, Y. Champoux, New empirical equations for sound propagation in rigid frame fibrous materials, J. Acoust. Soc. Am. 91 (6) (1992) 3346–3353 





[29] D. Lafarge, P. Lemarinier, J.F. Allard, V. Tarnow, Dynamic compressibility of air in porous structures at audible frequencies, J. Acoust. Soc. Am. 102 (4) (1997) 1995–2006. 





[30] D.K. Wilson, V.E. Ostashev, S.L. Collier, N.P. Symons, D.F. Aldridge, D.H. Marlin, Time-domain calculations of sound interactions with outdoor ground surfaces, Appl. Acoust. 68 (2) (2007) 173–200. 





[31] M. Fellah, Z.E.A. Fellah, E. Ogam, F. Mitri, C. Dépollier, Generalized equation for transient-wave propagation in continuous inhomogeneous rigid-frame porous materials at low frequencies, J. Acoust. Soc. Am. 134 (6) (2013) 4642–4647. 





[32] D.K. Wilson, V.E. Ostashev, S.L. Collier, Time-domain equations for sound propagation in rigid-frame porous media, J. Acoust. Soc. Am. 116 (4) (2004) 1889–1892. 





[33] O. Umnova, D. Turo, Time domain formulation of the equivalent fluid model for rigid porous media, J. Acoust. Soc. Am. 125 (4) (2009) 1860–1863 





[34] J. Zhao, M. Bao, X. Wang, H. Lee, S. Sakamoto, An equivalent fluid model based finite-difference time-domain algorithm for sound propagation in porous material with rigid frame, J. Acoust. Soc. Am. 143 (1) (2018) 130–138. 





[35] D. Dragna, P. Pineau, P. Blanc-Benon, A generalized recursive convolution method for time-domain propagation in porous media, J. Acoust. Soc. Am. 138 (2) (2015) 1030–1042. 





[36] I. Moufid, D. Matignon, R. Roncen, E. Piot, Energy analysis and discretization of the time-domain equivalent fluid model for wave propagation in rigid porous media, J. Comput. Phys. 451 (2022) 110888. 





[37] F. Pind, C.-H. Jeong, A.P. Engsig-Karup, J.S. Hesthaven, J. Strømann-Andersen, Time-domain room acoustic simulations with extended-reacting porous absorbers using the discontinuous Galerkin method, J. Acoust. Soc. Am. 148 (5) (2020) 2851–2863. 





[38] T. Yoshida, T. Okuzono, K. Sakagami, Time-domain finite element formulation of porous sound absorbers based on an equivalent fluid model, Acoust. Sci. Technol. 41 (6) (2020) 837–840. 





[39] A. Alomar, D. Dragna, M.-A. Galland, Time-domain simulations of sound propagation in a flow duct with extended-reacting liners, J. Sound Vib. 507 (2021) 116137. 





[40] J. Xie, M.Y. Ou, L. Xu, A discontinuous Galerkin method for wave propagation in orthotropic poroelastic media with memory terms, J. Comput. Phys. 397 (2019) 108865. 





[41] T. Bravo, C. Maury, Sound attenuation and absorption by micro-perforated panels backed by anisotropic fibrous materials: Theoretical and experimental study, J. Sound Vib. 425 (2018) 189–207. 





[42] K. Sakagami, S. Kobatake, K. Kano, M. Morimoto, M. Yairi, Sound absorption characteristics of a single microperforated panel absorber backed by a porous absorbent layer, Acoust. Australia 39 (3) (2011). 





[43] T. Okuzono, K. Uenishi, K. Sakagami, Experimental comparison of absorption characteristics of single-leaf permeable membrane absorbers with different backing air cavity designs, Noise Control Eng, J. 68 (3) (2020) 237–245. 





[44] M. Toyoda, J. Motooka, Prediction of permeable thin absorbers using the finite-difference time-domain method, J. Acoust. Soc. Am. 143 (5) (2018) 2870-2877 





[45] T. Okuzono, N. Shimizu, K. Sakagami, Predicting absorption characteristics of single-leaf permeable membrane absorbers using finite element method in a time domain, Appl., Acoust, 151 (2019) 172–182 





[46] S. Mukae, T. Okuzono, K. Tamaru. K. Sakagami, Modeling microperforated panels and permeable membranes for a room acoustic solver with plane-wave enriched fem, Appl. Acoust. 185 (2022) 108383. 





[47] T. Okuzono, K. Sakagami, A frequency domain finite element solver for acoustic simulations of 3D rooms with microperforated panel absorbers, Appl Acoust. 129 (2018) 1–12. 





[48] M. Toyoda, D. Eto, Prediction of microperforated panel absorbers using the finite-difference time-domain method, Wave Motion 86 (2019) 110–124. 





[49] T. Wu, C. Cheng, Z. Tao, Boundary element analysis of packed silencers with protective cloth and embedded thin surfaces, J. Sound Vib, 261 (1) (2003 1–15. 





[50] G. Gabard, O. Dazel, A discontinuous Galerkin method with plane waves for sound-absorbing materials, Internat. J. Numer. Methods Engrg. 104 (12) (2015) 1115–1138. 





[51] S. Wu, O. Dazel, G. Gabard, G. Legrain, High-order X-FEM for the simulation of sound absorbing poro-elastic materials with coupling interfaces, J. Sound Vib. 510 (2021) 116262. 





[52] J. Zhao, Z. Chen, M. Bao, S. Sakamoto, Prediction of sound absorption coefficients of acoustic wedges using finite-difference time-domain analysis, Appl. Acoust. 155 (2019) 428–441. 





[53] A. Alomar, D. Dragna, M.-A. Galland, Pole identification method to extract the equivalent fluid characteristics of general sound-absorbing materials, Appl. Acoust. 174 (2021) 107752. 





[54] B. Gustaysen, A. Semlven, Rational approximation of frequency domain responses by vector fitting, IEEE Trans. Power Deliv. 14 (3) (1999) 1052–1061. 





[55] R.M. Joseph, S.C. Hagness, A. Taflove, Direct time integration of Maxwell’s equations in linear dispersive media with absorption for scattering and propagation of femtosecond electromagnetic pulses, Opt. Lett. 16 (18) (1991) 1412–1414. 





[56] A.D. Pierce, Acoustics: An Introduction to Its Physical Principles and Applications, Springer International Publishing, Springer Nature, Switzerland AG 2019. 





[57] J.S. Hesthaven, T. Warburton, Nodal Discontinuous Galerkin Methods: Algorithms, Analysis and Applications, Springer-Verlag, New York, 2007. 





[58] H. Wang, I. Sihar, R. Pagán Muñoz, M. Hornikx, Room acoustics modelling in the time-domain with the nodal discontinuous Galerkin method, J. Acoust. Soc. Am. 145 (4) (2019) 2650–2663. 





[59] Shu C.-W., Discontinuous Galerkin method for time-dependent problems: Survey and recent developments, in: Recent Developments in Discontinuous Galerkin Finite Element Methods for Partial Differential Equations, 2014, pp. 25–62. 





[60] D.A. Kopriva, J. Nordström, G.J. Gassner, Error boundedness of discontinuous Galerkin spectral element approximations of hyperbolic problems, J. Sci. Comput. 72 (1) (2017) 314–330. 





[61] K. Duru, L. Rannabauer, A.-A. Gabriel, H. Igel, A new discontinuous Galerkin method for elastic waves with physically motivated numerical fluxes, J. Sci. Comput. 88 (3) (2021) 1–32. 





[62] M. Ainsworth, Dispersive and dissipative behaviour of high order discontinuous Galerkin finite element methods, J. Comput. Phys. 198 (1) (2004) 106–130 





[63] R.J. LeVeque, Finite Volume Methods for Hyperbolic Problems, Cambridge University Press, Cambridge, 2002 





[64] L.C. Wilcox, G. Stadler, C. Burstedde, O. Ghattas, A high-order discontinuous Galerkin method for wave propagation through coupled elastic–acoustic media, J. Comput. Phys. 229 (24) (2010) 9373–9396. 





[65] Q. Zhan, M. Zhuang, Y. Fang, Y. Hu, Y. Mao, W.-F. Huang, R. Zhang, D. Wang, Q.H. Liu, Full-anisotropic poroelastic wave modeling: A discontinuous Galerkin algorithm with a generalized wave impedance, Comput. Methods Appl. Mech. Engrg. 346 (2019) 288–311. 





[66] H.-O. Kreiss, Initial boundary value problems for hyperbolic systems, Comm. Pure Appl. Math. 23 (3) (1970) 277–298. 





[67] A. Majda, S. Osher, Initial–boundary value problems for hyperbolic equations with uniformly characteristic boundary, Comm. Pure Appl. Math. 28 (5) (1975) 607–675. 





[68] R.L. Higdon, Initial–boundary value problems for linear hyperbolic system, SIAM Rev. 28 (2) (1986) 177–217. 





[69] H. Wang, J. Yang, M. Hornikx, Frequency-dependent transmission boundary condition in the acoustic time-domain nodal discontinuous Galerkin model, Appl. Acoust. 164 (2020) 107280. 





[70] L.M. Brekhoyskikh, O.A. Godin, Acoustics of Lavered Media I: Plane and Ouasi-Plane Waves, Vol. 5. Springer Science & Business Media, 2012 





[71] D.-Y. Maa, Potential of microperforated panel absorber, J. Acoust. Soc. Am. 104 (5) (1998) 2861–2866. 





[72] H. Wang, M. Cosnefroy, M. Hornikx, An arbitrary high-order discontinuous Galerkin method with local time-stepping for linear acoustic wave propagation, J. Acoust. Soc. Am. 149 (1) (2021) 569–580. 





[73] B. Cockburn, C.-W. Shu, Runge-Kutta discontinuous Galerkin methods for convection-dominated problems, J. Sci. Comput, 16 (3) (2001) 173–261 





[74] B. Cotté, P. Blanc-Benon, C. Bogey, F. Poisson, Time-domain impedance boundary conditions for simulations of outdoor sound propagation, AIAA J. 47 (10) (2009) 2391–2403. 





[75] T. Toulorge, W. Desmet, Optimal Runge–Kutta schemes for discontinuous Galerkin space discretizations applied to wave propagation problems, J. Comput. Phys. 231 (4) (2012) 2067–2091. 





[76] J.-F. Allard, W. Lauriks, C. Verhaegen, The acoustic sound field above a porous layer and the estimation of the acoustic surface impedance from free-field measurements, J. Acoust. Soc. Am. 91 (5) (1992) 3057–3060. 





[77] C. Geuzaine, J.-F. Remacle, Gmsh: A 3-D finite element mesh generator with built-in pre-and post-processing facilities, Internat. J. Numer. Methods Engrg. 79 (11) (2009) 1309–1331. 

