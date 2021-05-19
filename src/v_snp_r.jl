module fun_v_snp_r

using GridInterpolations

import beta
import p_d
import xt_1 
import vvd
import vvr 
import x_grid 
import at_1 
import lambda 
import ecost_r

vr = interpolate(x_grid, vvr, x)
vd = interpolate(x_grid, vvd, x)

function v_snp_r(x) = -abs(xt_1 -x) + lambda * at_1 -ecost_r + beta *(vd* p_d + vr *(1 -p_d))

end