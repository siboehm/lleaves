
#include "header.h"

;


size_t get_num_class(void) {
  return 1;
}

size_t get_num_feature(void) {
  return 10;
}

const char* get_pred_transform(void) {
  return "identity";
}

float get_sigmoid_alpha(void) {
  return 1;
}

float get_global_bias(void) {
  return 0;
}

const char* get_threshold_type(void) {
  return "float64";
}

const char* get_leaf_output_type(void) {
  return "float64";
}


static inline double pred_transform(double margin) {
  return margin;
}
double predict(union Entry* data, int pred_margin) {
  double sum = (double)0;
  unsigned int tmp;
  int nid, cond, fid;  /* used for folded subtrees */
  if ( UNLIKELY( !(data[4].missing != -1) || (data[4].fvalue <= (double)0.1509544449887690043) ) ) {
    if ( LIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.6113719188800220694) ) ) {
      if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.5590042480032381134) ) ) {
        sum += (double)0.4952952376093183173;
      } else {
        sum += (double)0.4708363624215126109;
      }
    } else {
      sum += (double)0.5036285711910043483;
    }
  } else {
    if ( UNLIKELY( !(data[0].missing != -1) || (data[0].fvalue <= (double)0.2566254952377123311) ) ) {
      if ( UNLIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.342993610877122157) ) ) {
        sum += (double)0.5382810818762392424;
      } else {
        if ( UNLIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.4307520783814453025) ) ) {
          sum += (double)0.4995076919473134813;
        } else {
          if ( UNLIKELY( !(data[3].missing != -1) || (data[3].fvalue <= (double)0.4134798755216777533) ) ) {
            sum += (double)0.5122000000178813517;
          } else {
            sum += (double)0.5322000006139278039;
          }
        }
      }
    } else {
      if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.07270287079941535613) ) ) {
        sum += (double)0.4799272717833519275;
      } else {
        if ( UNLIKELY( !(data[5].missing != -1) || (data[5].fvalue <= (double)0.1287111408093345177) ) ) {
          sum += (double)0.5280333338230848161;
        } else {
          if ( LIKELY( !(data[7].missing != -1) || (data[7].fvalue <= (double)0.7788600780226724307) ) ) {
            sum += (double)0.5029711440214470475;
          } else {
            sum += (double)0.5185333335399627819;
          }
        }
      }
    }
  }

  sum = sum + (double)(0);
  if (!pred_margin) {
    return pred_transform(sum);
  } else {
    return sum;
  }
}
