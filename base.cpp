#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>

#include "base.h"

using namespace std; 

Base::Base(int aa, int bb) {
  a = aa;
  b = bb; 
}

Base::~Base() {
  cout << "Destroyed!\n"; 
} 

void Base::b_print() {
  cout << "a = " << a << ", b = " << b << "\n"; 
} 
