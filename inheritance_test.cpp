#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "derived.h"

int main() {
  Derived derived = new Derived(1,2,3);

  derived -> b_print(); 
  derived -> print(); 
}
