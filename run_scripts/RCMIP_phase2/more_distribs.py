import math
import warnings
import numpy as np
import scipy.stats as st
import scipy.special as sc
import matplotlib.pyplot as plt

from scipy.optimize import fmin
from scipy.integrate import quad
from scipy._lib._util import _lazywhere


##################################################
## NEW DISTRIBS
##################################################

## type-2 generalized normal distribution class
class gen_norm2_gen(st.rv_continuous):

    def _argcheck(self, k):
        return np.isfinite(k)

    def _pdf(self, x, k):
        h = 1E-6
        return (self._cdf(x + h, k) - self._cdf(x, k)) / h

    def _cdf(self, x, k):
        y = _lazywhere(k == 0, [x, k],
            lambda x_, k_: x_ / k_,
            f2 = lambda x_, k_: -1 / k_ * np.log(1 - k_ * x_))
        return np.nan_to_num(st._continuous_distns._norm_cdf(y)) # dirty!


## extended skew-normal distribution class
class extskew_norm_gen(st.rv_continuous):

    def _argcheck(self, a, tau):
        return np.isfinite(a) & np.isfinite(tau)

    def _pdf(self, x, a, tau):
        return st._continuous_distns._norm_pdf(x) * st._continuous_distns._norm_cdf(tau * np.sqrt(1 + a**2) + a*x) / st._continuous_distns._norm_cdf(tau)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        cdf = quad(self._pdf, _a, x, args=args)[0]
        if cdf > 1: cdf = 1.0
        return cdf


## skew-generalized normal distribution class
class skewgen_norm_gen(st.rv_continuous):

    def _argcheck(self, a1, a2):
        return np.isfinite(a1) & (a2 >= 0)

    def _pdf(self, x, a1, a2):
        return 2. * st._continuous_distns._norm_pdf(x) * st._continuous_distns._norm_cdf(a1 * x / np.sqrt(1 + a2 * x**2))

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        cdf = quad(self._pdf, _a, x, args=args)[0]
        if cdf > 1: cdf = 1.0
        return cdf


## flexible generalized skew-normal distribution class
## with 3rd degree polynomial
class flexgenskew3_norm_gen(st.rv_continuous):

    def _argcheck(self, a1, a3):
        return np.isfinite(a1) & np.isfinite(a3)

    def _pdf(self, x, a1, a3):
        return 2. * st._continuous_distns._norm_pdf(x) * st._continuous_distns._norm_cdf(a1 * x + a3 * x**3)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        cdf = quad(self._pdf, _a, x, args=args)[0]
        if cdf > 1: cdf = 1.0
        return cdf


## with 5th degree polynomial
class flexgenskew5_norm_gen(st.rv_continuous):

    def _argcheck(self, a1, a3, a5):
        return np.isfinite(a1) & np.isfinite(a3) & np.isfinite(a5)

    def _pdf(self, x, a1, a3, a5):
        return 2. * st._continuous_distns._norm_pdf(x) * st._continuous_distns._norm_cdf(a1 * x + a3 * x**3 + a5 * x**5)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        cdf = quad(self._pdf, _a, x, args=args)[0]
        if cdf > 1: cdf = 1.0
        return cdf


## generate distributions
st_gennorm2 = gen_norm2_gen(name='gennorm2')
st_extskewnorm = extskew_norm_gen(name='extskewnorm')
st_skewgennorm = skewgen_norm_gen(name='skewgennorm')
st_flexgenskewnorm3 = flexgenskew3_norm_gen(name='flexgenskewnorm3')
st_flexgenskewnorm5 = flexgenskew5_norm_gen(name='flexgenskewnorm5')


##################################################
## LIST DISTRIBS
##################################################

## list distribs
distrib_all = [
st.alpha,
st.anglit,
st.arcsine,
st.argus,
st.beta,
st.betaprime,
st.bradford,
st.burr,
st.burr12,
st.cauchy,
st.chi,
st.chi2,
st.cosine,
st.crystalball,
st.dgamma,
st.dweibull,
# st.erlang, # integer param
st.expon,
st.exponnorm,
st.exponweib,
st.exponpow,
st.f,
st.fatiguelife,
st.fisk,
st.foldcauchy,
st.foldnorm,
#st.frechet_r, # deprecated
#st.frechet_l, # deprecated
st.genlogistic,
st.gennorm,
st.genpareto,
st.genexpon,
st.genextreme,
st.gausshyper,
st.gamma,
st.gengamma,
st.genhalflogistic,
# st.geninvgauss, # long!
st.gilbrat,
st.gompertz,
st.gumbel_r,
st.gumbel_l,
st.halfcauchy,
st.halflogistic,
st.halfnorm,
st.halfgennorm,
st.hypsecant,
st.invgamma,
st.invgauss,
st.invweibull,
st.johnsonsb,
st.johnsonsu,
st.kappa4,
st.kappa3,
st.ksone,
#st.kstwo, # requires update
st.kstwobign,
st.laplace,
st.levy,
st.levy_l,
st.levy_stable,
st.logistic,
st.loggamma,
st.loglaplace,
st.lognorm,
#st.loguniform, # update for me?
st.lomax,
st.maxwell,
st.mielke,
st.moyal,
st.nakagami,
st.ncx2,
st.ncf,
st.nct,
st.norm,
st.norminvgauss,
st.pareto,
st.pearson3,
st.powerlaw,
st.powerlognorm,
st.powernorm,
st.rdist,
st.rayleigh,
st.rice,
st.recipinvgauss,
st.semicircular,
st.skewnorm,
st.t,
st.trapz,
st.triang,
st.truncexpon,
st.truncnorm,
st.tukeylambda,
st.uniform,
st.vonmises,
st.vonmises_line,
st.wald,
st.weibull_min,
st.weibull_max,
st.wrapcauchy,
]

## only with infinite support
distrib_inf = [dist for dist in distrib_all if dist.a == -np.inf and dist.b == np.inf]

## new distribs
distrib_new = [st_gennorm2, st_extskewnorm, st_skewgennorm, st_flexgenskewnorm3, st_flexgenskewnorm5]


##################################################
## FIT DISTRIBS
##################################################

## try all distribs
def try_all_distrib(xy_list, x_val, distrib_list=distrib_all+distrib_new, rtol=0.00, atol=0.01, first_guess=1.):

    distrib_out = []
    for distrib in distrib_list:
        print(distrib, end='\r')

        def err(param):
            dist = 0.
            for xy in xy_list:
                dist += (distrib.cdf(xy[0], *param) - xy[1])**2
            return dist

        is_close = 5*[False]
        fmin_flag = None
        try:
            param, _, _, _, fmin_flag = fmin(err, distrib.numargs*[first_guess] + [x_val[0], x_val[1]], xtol=1E-9, ftol=1E-9, full_output=True, disp=False)
            is_close = np.isclose(distrib.cdf([x for x,y in xy_list], *param), [y for x,y in xy_list], rtol=rtol, atol=atol)
        except: 
            pass
        
        if fmin_flag==0 and np.where(np.isnan([x for x,y in xy_list]), 5*[True], is_close).all():
            distrib_out.append((distrib, param))

    return distrib_out

## use it on ECS
if __name__ == '__main__':
    zou = try_all_distrib([(2, 0.05), (2.5, 0.17), (3, 0.50), (4, 0.83), (5, 0.95)], [3, (4-2.5)/2.], atol=0.02)
    #zou = try_all_distrib([(2, 0.05), (2.5, 0.17), (3, 0.50), (4, 0.83), (5, 0.95)], [3, (4-2.5)/2.], distrib_list=[st_gennorm2], atol=0.05, first_guess=0.)

    plt.figure()
    plt.plot([2, 2.5, 3, 4, 5], [0.05, 0.17, 0.50, 0.83, 0.95], ls='none', marker='+', ms=8, mew=3)
    for distrib, param in zou:
        print(distrib.name, distrib.cdf([2, 2.5, 3, 4, 5], *param))
        plt.plot(np.arange(-1,10,0.1), distrib.cdf(np.arange(-1,10,0.1), *param), label=distrib.name)
    plt.legend(loc=0)
    plt.show()

