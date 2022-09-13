#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _CaDynamics_reg();
extern void _Ca_HVA_reg();
extern void _Ca_LVA_reg();
extern void _Ih_reg();
extern void _Im_reg();
extern void _Im_v2_reg();
extern void _K_P_reg();
extern void _K_T_reg();
extern void _Kd_reg();
extern void _Kv2like_reg();
extern void _Kv3_1_reg();
extern void _NaTa_reg();
extern void _NaTs_reg();
extern void _NaV_reg();
extern void _Nap_reg();
extern void _SK_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," CaDynamics.mod");
fprintf(stderr," Ca_HVA.mod");
fprintf(stderr," Ca_LVA.mod");
fprintf(stderr," Ih.mod");
fprintf(stderr," Im.mod");
fprintf(stderr," Im_v2.mod");
fprintf(stderr," K_P.mod");
fprintf(stderr," K_T.mod");
fprintf(stderr," Kd.mod");
fprintf(stderr," Kv2like.mod");
fprintf(stderr," Kv3_1.mod");
fprintf(stderr," NaTa.mod");
fprintf(stderr," NaTs.mod");
fprintf(stderr," NaV.mod");
fprintf(stderr," Nap.mod");
fprintf(stderr," SK.mod");
fprintf(stderr, "\n");
    }
_CaDynamics_reg();
_Ca_HVA_reg();
_Ca_LVA_reg();
_Ih_reg();
_Im_reg();
_Im_v2_reg();
_K_P_reg();
_K_T_reg();
_Kd_reg();
_Kv2like_reg();
_Kv3_1_reg();
_NaTa_reg();
_NaTs_reg();
_NaV_reg();
_Nap_reg();
_SK_reg();
}
