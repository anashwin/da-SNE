#ifndef DERIVED
#define DERIVED
class Derived : public Base {
 private:
  int c;
 public:
 Derived(int aa, int bb, int cc) : Base(aa, bb);
  ~Derived();
  void print(); 

}; 
#endif
