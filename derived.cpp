#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>

#include "derived.h"

Derived::Derived(int aa, int bb, int cc) : public Base::Base(aa, bb) {
  c = cc; 
} 

Derived::~Derived() {
  cout << "Derived destroyed!\n"; 
} 

void Derived::print() {
  cout << "a = " << a << ", b = " << b << ", c = " << c << "\n"; 
} 
