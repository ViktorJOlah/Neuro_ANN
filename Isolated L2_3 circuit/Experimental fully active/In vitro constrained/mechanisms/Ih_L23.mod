: Ih_L23: Kole, Hallermann and Stuart (2006) HCN kinetics (Allen Ih.mod) with two added RANGE
: parameters for the Experimental fully active L2/3 model: vshift (mV, shifts both rate curves)
: and tauscale (divides tau; >1 = faster, HCN1-like). Defaults reproduce Ih.mod.

NEURON	{
	SUFFIX Ih_L23
	NONSPECIFIC_CURRENT ihcn
	RANGE gbar, g, ihcn, vshift, tauscale, ehcn 
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2) 
	ehcn =  -45.0 (mV)
	vshift = 0 (mV)
	tauscale = 1
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m
	ihcn = g*(v-ehcn)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
    :    if(v == -154.9){
    :       v = v + 0.0001
    :    }
		:mAlpha =  0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
		mAlpha = 0.001 * 6.43 * vtrap(v - vshift + 154.9, 11.9)
		mBeta  =  0.001*193*exp((v - vshift)/33.1)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)/tauscale
	UNITSON
}

FUNCTION vtrap(x, y) { : Traps for 0 in denominator of rate equations
	UNITSOFF
	if (fabs(x / y) < 1e-6) {
		vtrap = y * (1 - x / y / 2)
	} else {
		vtrap = x / (exp(x / y) - 1)
	}
	UNITSON
}
