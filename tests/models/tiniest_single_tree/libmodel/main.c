
#include "header.h"

;


size_t get_num_class(void) {
  return 1;
}

size_t get_num_feature(void) {
  return 3;
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
  if ( LIKELY( !(data[1].missing != -1) || (data[1].fvalue <= (double)0.7314494145219632149) ) ) {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.9070836626874522191) ) ) {
      sum += (double)0.4951066126651433863;
    } else {
      sum += (double)0.5068894836955886163;
    }
  } else {
    if ( LIKELY( !(data[2].missing != -1) || (data[2].fvalue <= (double)0.8555160147839011575) ) ) {
      sum += (double)0.5064538220029983773;
    } else {
      sum += (double)0.4904060235782387589;
    }
  }

  sum = sum + (double)(0);
  if (!pred_margin) {
    return pred_transform(sum);
  } else {
    return sum;
  }
}
