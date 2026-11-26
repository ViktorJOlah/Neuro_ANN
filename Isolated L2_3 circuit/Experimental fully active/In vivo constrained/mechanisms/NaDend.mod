:NaDend: NaTs2_t kinetics (Colbert & Pan 2002, as in the Allen models) with separate activation and
:inactivation shifts (vshm, vshh, mV; positive = depolarised). Used for the dendritic Na+ hot spots that
:generate local spikelets on NMDA/Ca2+ plateaus (Smith et al. 2013 in vivo constraint).
:Reference :Colbert and Pan 2002
:comment: took the NaTa and shifted both activation/inactivation by 6 mv

NEURON	{
	SUFFIX NaDend
	USEION na READ ena WRITE ina
	RANGE gbar, g, ina, vshm, vshh
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	vshm = 0 (mV)
	vshh = 0 (mV)
}

ASSIGNED	{
	v	(mV)
	ena	(mV)
	ina	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
	hInf
	hTau
	hAlpha
	hBeta
}

STATE	{
	m
	h
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m*m*m*h
	ina = g*(v-ena)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
	h' = (hInf-h)/hTau
}

INITIAL{
	rates()
	m = mInf
	h = hInf
}

PROCEDURE rates(){
  LOCAL qt, vm, vh
  qt = 2.3^((34-21)/10)

	UNITSOFF
    vm = v - vshm
    if(vm == -32){
    	vm = vm+0.0001
    }
		mAlpha = (0.182 * (vm- -32))/(1-(exp(-(vm- -32)/6)))
		mBeta  = (0.124 * (-vm -32))/(1-(exp(-(-vm -32)/6)))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = (1/(mAlpha + mBeta))/qt

    vh = v - vshh
    if(vh == -60){
      vh = vh + 0.0001
    }
		hAlpha = (-0.015 * (vh- -60))/(1-(exp((vh- -60)/6)))
		hBeta  = (-0.015 * (-vh -60))/(1-(exp((-vh -60)/6)))
		hInf = hAlpha/(hAlpha + hBeta)
		hTau = (1/(hAlpha + hBeta))/qt
	UNITSON
}
