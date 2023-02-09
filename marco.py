# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from z3 import *
import time
import z3.z3util
from itertools import combinations


def set_core_minimize(s):
    s.set(':core.minimize', True)


'''
time_cost = list()
ass_len = list()
MUC_len = list()
res = open("res.txt",'w')
k = 4

while k < 12:
    s = Solver()
    s.from_file("./FAT" + str(k) + ".txt")
    print("Solver1===========")
    #print(s.sexpr())
    print(s.check())
    #c = s.unsat_core()
    #print(len(c))
    #print(c)
    print("===========")
    assertions = s.assertions()
    #print("Assertion:")
    #print(z3util.get_vars(assertions[0]))
    #print(len(assertions))
    i = 0
    #print(assertions[len(assertions) - 1])

    print("Solver2:==========")
    p_dict = {}         #save the asserts of unsat_core
    p_tag = list()     #save the tags of unsat_core
    MUC = list()       #save the line number of MUC

    s2 = Solver()
    s2.set(unsat_core=True)
    set_core_minimize(s2)         #minimal unsat core
    while i < len(assertions):
        s2.assert_and_track(assertions[i], Bool("p" + str(i)))
        i = i + 1
    #print(s2.sexpr())
    print(s2.check())
    time_start = time.time()
    s2_core = s2.unsat_core()
    time_end = time.time()
    res.write("time cost: " + '\n' + str(time_end - time_start) + '\n')
    time_cost.append(time_end - time_start)
    res.write("unsat_core size:" + '\n')
    res.write(str(len(s2_core)) + '\n')
    MUC_len.append(len(s2_core))
    res.write("SMT file size:" + '\n')
    res.write(str(len(assertions)) + '\n')
    ass_len.append(len(assertions))
    k += 2
'''

'''
##Get assersions of minimal? unsat_core and save in a solver
s3 = Solver()
i = 0
while i < len(assertions):
    if Bool("p" + str(i)) in s2_core:
        s3.assert_and_track(assertions[i], Bool("p" + str(i)))
        #print(str(assertions[i]) + "   " + str(i))
        MUC.append(i + 1)
        p_tag.append(Bool("p" + str(i)))
        p_dict[Bool("p" + str(i))] = assertions[i]
    i = i + 1

print("Solver3:")
print(s3.check())

print("MUC:")
print(MUC)
print(p_tag)
'''

'''
print(s3_core)
print("unsat_core assertion number:")
print(len(s3_core))
'''
'''
print("p_tag:")
print(p_tag)
print(p_dict)


print("Check if it is an minimal unsat_core:")
s3_assertions = s3.assertions()
i = 1
tag = True
'''

'''
while i < len(p_tag):
    combs = combinations(p_tag, i)
    #print(comb)
    for comb in list(combs):  # one combination
        s4 = Solver()
        for item in comb:
            #print(p_dict[item])
            s4.assert_and_track(p_dict[item], item)
        if s4.check() == unsat:
            tag = False
            #print("there is an subset of unsat_core that is unsat.")
            print(s4)
        #else:
            #print("sat")
    i = i + 1

if tag == False:
    print("there is an subset of unsat_core that is unsat")
'''
'''
combs = combinations(p_tag, i)
for comb in list(combs):  # one combination
    s4 = Solver()
    for item in comb:
        s4.assert_and_track(p_dict[item], item)
    if s4.check() == unsat:
        tag = False
        print(s4)
    #else:
        #print("sat")

if tag == False:
    print("there is an subset of unsat_core that is unsat")
else:
    print("It is a minimal unsat_core")
'''
'''
print("MUC:")
print(MUC)
print("s2 unsat_core")
print(s2_core)
print("The size of unsat core:")
print(len(s2_core))
'''

# unsat core example
'''
p1, p2, p3 = Bools('p1 p2 p3')
x, y = Ints('x y')
s = Solver()
s.add(Implies(p1, x > 0))
s.add(Implies(p2, y > x))
s.add(Implies(p2, y < 1))
s.add(Implies(p3, y > -3))
#s.assert_and_track(('assert (<= |0_FAILED-NODE_montreal| 1))', 'p1')
print(s.sexpr())
print(s.check(p1, p2, p3))
core = s.unsat_core()
print(core)
print(len(core))
print(p1 in core)
print(p2 in core)
print(p3 in core)
'''

'''
p, q, r, v = Bools('p q r v')
s = Solver()
s.add(Not(q))
s.add(Implies(p,q))
s.add(p)
#s.set("sat.core.minimize","true")
print(s.sexpr())
print(s.check())
print(s.unsat_core())
'''

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# new_filepath = './canerie_notexport.txt'
'''
with open('./canerie_notexport.txt', 'r') as file:
    lines = file.readlines()
    text = ""
    for line in lines:
        text += line
'''
'''
def tt(s, f):
    return is_true(s.model().eval(f))

def get_mss(s, ps):
    if sat != s.check():
       return []
    mss = { q for q in ps if tt(s, q) }
    return get_mss(s, mss, ps)

def get_mss(s, mss, ps):
    ps = ps - mss
    backbones = set([])
    while len(ps) > 0:
       p = ps.pop()
       if sat == s.check(mss | backbones | { p }):
          mss = mss | { p } | { q for q in ps if tt(s, q) }
          ps  = ps - mss
       else:
          backbones = backbones | { Not(p) }
    return mss


def ff(s, p):
    return is_false(s.model().eval(p))


def marco(s, ps):
    print("The begin of marco:")
    map = Solver()
    set_core_minimize(s)
    while map.check() == sat:
        seed = {p for p in ps if not ff(map, p)}
        if s.check(seed) == sat:
            mss = get_mss(s, seed, ps)
            map.add(Or(ps - mss))
            yield "MSS", mss
        else:
            mus = s.unsat_core()
            map.add(Not(And(mus)))
            yield "MUS", mus
    print("======")

k = 4
s = Solver()
s.from_file("./FAT" + str(k) + ".txt")
print("Solver1===========")
#print(s.sexpr())
print(s.check())
#c = s.unsat_core()
#print(len(c))
#print(c)
print("===========")
ass = s.assertions()
#print("Assertion:")
#print(z3util.get_vars(assertions[0]))
#print(len(assertions))
i = 0
#print(assertions[len(assertions) - 1])

print("Solver2:==========")
p_dict = {}         #save the asserts of unsat_core
p_tag = list()     #save the tags of unsat_core
MUC = list()       #save the line number of MUC

s2 = Solver()
s2.set(unsat_core=True)
set_core_minimize(s2)         #minimal unsat core
while i < len(ass):
    s2.assert_and_track(ass[i], Bool("p" + str(i)))
    i = i + 1
    #print(s2.sexpr())
print(s2.check())
time_start = time.time()
s2_core = s2.unsat_core()
time_end = time.time()
s3 = Solver()
print("Solver3:")
marco(s3, s2.assertions())
print("marco finish")
'''

# k = 4
# num1 = 0
def main():
    if len(sys.argv) < 2:
        print("input the smt file path.")
    smt_file_path = sys.argv[1]
    s = Solver()
    s.from_file(smt_file_path)
    print("===========")
    # print(s.sexpr())
    print(s.check())
    # c = s.unsat_core()
    # print(len(c))
    # print(c)
    print("===========")
    if s.check() == z3.sat:
        print("The smt file is SAT.")
        return
    ass = s.assertions()
    csolver = SubsetSolver(ass)
    msolver = MapSolver(n=csolver.n)
    for orig, lits in enumerate_sets(csolver, msolver):
        output = "%s %s" % (orig, lits)
        #num1 = num1 + 1
        #print(output)
        if orig == "MUS":
            print(output)
def get_id(x):
    return Z3_get_ast_id(x.ctx.ref(), x.as_ast())


def MkOr(clause):
    if clause == []:
        return False
    else:
        return Or(clause)


class SubsetSolver:
    constraints = []
    n = 0      # the length of constriant
    s = Solver()
    varcache = {}
    idcache = {}

    def __init__(self, constraints):
        self.constraints = constraints
        self.n = len(constraints)
        for i in range(self.n):
            self.s.add(Implies(self.c_var(i), constraints[i]))
            # print(Implies(self.c_var(i), constraints[i]))
        print("")
    def c_var(self, i):
        if i not in self.varcache:
            v = Bool(str(self.constraints[abs(i)]))
            self.idcache[get_id(v)] = abs(i)
            if i >= 0:
                self.varcache[i] = v
            else:
                self.varcache[i] = Not(v)
        return self.varcache[i]

    def check_subset(self, seed):
        assumptions = self.to_c_lits(seed)
        return (self.s.check(assumptions) == sat)

    def to_c_lits(self, seed):
        return [self.c_var(i) for i in seed]

    def complement(self, aset):
        return set(range(self.n)).difference(aset)

    def seed_from_core(self):
        core = self.s.unsat_core()
        return [self.idcache[get_id(x)] for x in core]

    def shrink(self, seed):
        current = set(seed)
        for i in seed:
            if i not in current:
                continue
            current.remove(i)
            if not self.check_subset(current):         #unsat
                current = set(self.seed_from_core())
            else:
                current.add(i)
        return current

    def grow(self, seed):
        current = seed
        for i in self.complement(current):
            current.append(i)
            if not self.check_subset(current):
                current.pop()
        return current


class MapSolver:
    def __init__(self, n):
        """Initialization.
           Args:
            n: The number of constraints to map.
        """
        self.solver = Solver()
        self.n = n
        self.all_n = set(range(n))  # used in complement fairly frequently

    def next_seed(self):
        """Get the seed from the current model, if there is one.
            Returns:
            A seed as an array of 0-based constraint indexes.
        """
        if self.solver.check() == unsat:
            return None
        seed = self.all_n.copy()  # default to all True for "high bias"
        model = self.solver.model()
        for x in model:
            if is_false(model[x]):
                seed.remove(int(x.name()))
        return list(seed)

    def complement(self, aset):
        """Return the complement of a given set w.r.t. the set of mapped constraints."""
        return self.all_n.difference(aset)

    def block_down(self, frompoint):
        """Block down from a given set."""
        comp = self.complement(frompoint)
        self.solver.add(MkOr([Bool(str(i)) for i in comp]))

    def block_up(self, frompoint):
        """Block up from a given set."""
        self.solver.add(MkOr([Not(Bool(str(i))) for i in frompoint]))


def enumerate_sets(csolver, map):
    """Basic MUS/MCS enumeration, as a simple example."""
    while True:
        seed = map.next_seed()
        if seed is None:
            return
        if csolver.check_subset(seed):         #sat
            '''
            MSS = csolver.grow(seed)
            yield ("MSS", csolver.to_c_lits(MSS))
            map.block_down(MSS)
            '''
        else:             #unsat
            MUS = csolver.shrink(seed)
            yield ("MUS", MUS)
            map.block_up(MUS)


main()
print("muc num")
print(num1)