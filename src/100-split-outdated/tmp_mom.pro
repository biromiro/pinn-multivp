;; some pars on the fly
i = 11
j= 639

xr=[2, 31]

;; units
DX_UN  =   6.96000e+10  ;; length unit factor (to cm)
VX_UN  =   1.28557e+07  ;; speed unit factor (to cm/s)
TE_UN  =   1.00000e+06  ;; temperature unit factor (to K)
NE_UN  =   1.00000e+17  ;; density unit factor (to 1/cm^3)
DT_UN  =       5413.93  ;; time unit factor (to s)

nu_un = dx_un^2/dt_un   ;; viscosity coeff init factor (to cm^2/s)

nuvisc = .1 * nu_un    ;; the actual viscosity coeff

;; read MVP
ll  = tubes[i].x_i_[0:j]       ;; ll, x_i_: curvilinear coordinate
len = ll * dx_un               ;; len: curvilinear coordinate in CGS
rad = tubes[i].xradial[0:j] * dx_un ;; xradial: radial coordinate in code units; rad: same in CGS
vel = tubes[i].u[0:j] * vx_un       ;; u: speed in code units; vel: speed in CGS
den = tubes[i].n[0:j] * ne_un       ;; n: density in code units; den: speed in CGS
tem = tubes[i].t__mk_[0:j] * te_un  ;; t__mk_: temperature in code units (MK); tem temperature in CGS
alf = tubes[i].inclin[0:j]          ;; inclin, alf: inclination angle alpha
aprimea = tubes[i].expans[0:j] / dx_un ;; expans: A'/A in code units; aprimea: same in CGS (has dims of 1/L)


;; grad(pressure) term
gradp  = deriv(len,den*tem)/den

;; gravitational term:
gravf  = grav*cos(alf)/rad^2

;; flow expansion term (v.grad)v, expanded into 4 sub-terms
vgradva  = deriv(len,vel^2)
vgradvb  = aprimea*vel^2
vgradvc  = - vel*deriv(len,vel)
vgradvd  = - aprimea*vel^2
;;vgradv = vgradva + vgradvb + vgradvc + vgradvd
;; terms B and Dcancel out:
vgradv = vgradva + vgradvc 

;; viscous term (expanded in two terms)
visca = deriv(len,deriv(len, vel))
viscb = deriv(len,vel)*aprimea

visc  = -nuvisc*(visca+viscb)

;; all together (should ~0)
sum = vgradv + gradp + gravf + visc

;; plots
x2
plot, ll, vgradv, tit='V grad V', /xsty, xr=xr, xtit='R [R!Dsun!N]'
plot, ll, gradp , tit='grad P / n', /xsty, xr=xr, xtit='R [R!Dsun!N]'
;;plot, ll, gravf , tit='n*g', /xsty, xr=xr, xtit='R [R!Dsun!N]'
plot, ll, visc , tit='nu*lap(v)', /xsty, xr=xr, xtit='R [R!Dsun!N]'
plot, ll, sum, tit='sum  (should be ~0)', /xsty, xr=xr, xtit='R [R!Dsun!N]'



end
