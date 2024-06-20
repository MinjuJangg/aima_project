from logic import * #logic 파일을 사용하기 위해 import 함
import itertools

class KB:  # 지식 베이스 클래스
    """신규 문장을 추가(tell)하거나 알려진 것을 질의(ask)할 수 있는 지식베이스(knowledge base).
    지식베이스를 생성하려면 이 클래스의 서브클래스로 정의하고 tell, ask_generator, retract 등을 구현하면 됨.
    ask_generator는 문장이 참이 되도록 하는 대입들을 찾고, ask는 이 중 첫번째를 리턴하거나 False 리턴."""

    def __init__(self, sentence=None): #KB클래스의 생성자로 객체가 생성될 때 호출되며 sentence를 tell 함수를 이용하여 지식베이스에 추가한다.
        if sentence:
            self.tell(sentence)

    def tell(self, sentence): #지식베이스에 신규 문장을 업데이트 할 때 사용하는 tell 함수이다. 
        """지식베이스에 문장 추가"""
        raise NotImplementedError

    def ask(self, query): #지식 베이스에 알려진 것을 질의하는 ask 함수이다.
        """query를 참이 되게 하는 (첫번째) 대입을 리턴함. 없으면 False 리턴."""
        return first(self.ask_generator(query), default=False)

    def ask_generator(self, query): #query가 참이 되기 위한 변수들에 대해 가능한 모든 값들의 조합을 생성하는 함수이다.
        """query가 참이 되는 모든 대입들을 생성"""
        raise NotImplementedError

    def retract(self, sentence): #지식베이스에서 특정 문장을 제거하는 함수이다.
        """지식베이스에서 문장 삭제"""
        raise NotImplementedError
    

class FolKB(KB):  #일차논리 한정절에 관련한 작업을 위한 지식베이스를 나타내는 클래스
    """일차논리 한정 절(definite clause)로 구성된 지식베이스.
    >>> kb0 = FolKB([expr('Farmer(Mac)'), expr('Rabbit(Pete)'),
    ...              expr('(Rabbit(r) & Farmer(f)) ==> Hates(f, r)')])
    >>> kb0.tell(expr('Rabbit(Flopsie)'))
    >>> kb0.retract(expr('Rabbit(Pete)'))
    >>> kb0.ask(expr('Hates(Mac, x)'))[x]
    Flopsie
    >>> kb0.ask(expr('Wife(Pete, x)'))
    False
    """

    def __init__(self, clauses=None): #FolKB 클래스의 생성자로 각 절이 주어졌을 때 tell 함수를 이용해서 KB에 업데이트한다.
        super().__init__()
        self.clauses = [] #self.clauses 리스트 생성 및 초기화
        if clauses:
            for clause in clauses:
                self.tell(clause)

    def tell(self, sentence): #sentence를 KB에 추가한다. 
        if is_definite_clause(sentence): #추가하는 조건은 문장이 일차논리 한정절인지 확인하고, 맞으면 KB에 추가하고,
            self.clauses.append(sentence)
        else:
            raise Exception(f'Not a definite clause: {sentence}') #아니면 일차논리 한정절이 아니라는 오류를 띄운다.

    def ask_generator(self, query): #query에 대해 참이 되는 모든 대입들을 생성한다.
        return fol_bc_ask(self, query) #fol_bc_ask 함수를 통해 query에 대한 대입들을 생성하는 제너레이터를 반환한다.

    def retract(self, sentence): #KB에서 sentence를 제거한다.
        self.clauses.remove(sentence)

    def fetch_rules_for_goal(self, goal): #목표에 적용 가능한 절들을 모두 반환한다.
        return self.clauses #self.clauses 리스트를 반환하게 된다.


def is_definite_clause(s): #주어진 s가 일차논리 한정절인지 확인하는 함수이다.
    """Expr s가 한정 절이면 True를 리턴함.
    A & B & ... & C ==> D  (모두 양 리터럴)
    절 형식으로 표현하면,
    ~A | ~B | ... | ~C | D   (하나의 양 리터럴을 갖는 절)
    >>> is_definite_clause(expr('Farmer(Mac)'))
    True
    """
    if is_symbol(s.op): #일차논리 한정절의 형식을 만족하는지 확인하면서 맞으면 True, 아니면 False를 반환한다.
        return True
    elif s.op == '==>':
        antecedent, consequent = s.args
        return is_symbol(consequent.op) and all(is_symbol(arg.op) for arg in conjuncts(antecedent))
    else:
        return False

def parse_definite_clause(s):  #주어진 일차논리 한정절을 전제와 결론으로 분리하여 반환하는 함수이다.
    """한정 절의 전제와 결론을 리턴"""
    assert is_definite_clause(s)
    if is_symbol(s.op):
        return [], s
    else:
        antecedent, consequent = s.args
        return conjuncts(antecedent), consequent

def conjuncts(s): #주어진 일차논리 한정절이 논리곱으로 해석 가능할 때만 리스트의 형태로 반환한다.
    """문장 s를 논리곱으로 해석 했을 때의 구성요소를 리스트로 리턴함.
    >>> conjuncts(A & B)
    [A, B]
    >>> conjuncts(A | B)
    [(A | B)]
    """
    return dissociate('&', [s])

def disjuncts(s): #주어진 일차논리 한정절이 논리합으로 해석 가능할 때만 리스트의 형태로 반환한다.
    """문장 s를 논리합으로 해석했을 때의 구성요소를 리스트로 리턴함.
    >>> disjuncts(A | B)
    [A, B]
    >>> disjuncts(A & B)
    [(A & B)]
    """
    return dissociate('|', [s])

def dissociate(op, args): #주어진 연산자에 따라 구성요소들을 리스트에 넣어서 반환한다.
    """op를 기준으로 인자들의 리스트를 리턴.
    >>> dissociate('&', [A & B])
    [A, B]
    """
    result = []

    def collect(subargs): #subargs는 주어진 인자들인데 각 인자를 확인하면서 주어진 연산자와 같은 인자를 발견하면 다시 cllect 함수에 전달 해 재귀적으로 수행한다.
        for arg in subargs:
            if arg.op == op:
                collect(arg.args)
            else:
                result.append(arg)

    collect(args)
    return result

def fol_fc_ask(kb, alpha): # kb를 바탕으로 alpha 문장이 참인지
                           # 순방향 연쇄 알고리즘을 사용해서 KB에 있는 문장들을 바탕으로 alpha 문장이 참인지를 판단하는 함수이다.
    """순방향 연쇄(forward chaining) 알고리즘"""
    """순방향 연쇄(forward chaining) 알고리즘"""
    kb_consts = list({c for clause in kb.clauses for c in constant_symbols(clause)})

    def enum_subst(p): #enum_subst 함수는 문장에 있는 변수들에 가능한 모든 대입을 생성한다. 
        query_vars = list({v for clause in p for v in variables(clause)})
        for assignment_list in itertools.product(kb_consts, repeat=len(query_vars)):
            theta = {x: y for x, y in zip(query_vars, assignment_list)}
            yield theta

    # 새로운 추론 없이도 답변할 수 있는지 체크
    for q in kb.clauses:
        phi = unify_mm(q, alpha)
        if phi is not None:
            yield phi

    while True:
        new = []
        for rule in kb.clauses:
            p, q = parse_definite_clause(rule)
            for theta in enum_subst(p):
                if set(subst(theta, p)).issubset(set(kb.clauses)):
                    q_ = subst(theta, q)
                    if all([unify_mm(x, q_) is None for x in kb.clauses + new]):
                        new.append(q_)
                        phi = unify_mm(q_, alpha)
                        if phi is not None:
                            yield phi
        if not new:
            break
        for clause in new:
            kb.tell(clause)
    return None

# 현재 KB에 있는 모든 문장들에 p를 대입했을 때 결론 q를 도출해서 새로운 문장을 만들어낸다.
# 새로운 문장이 생성되면 alpha 문장과 비교해서 단일화가 가능한지 확인한다. phi에 대입
# 단일화가 가능한 경우, alpha 문장이 참이되는 단일화인 phi를 반환한다.

def unify_mm(x, y, s={}): #unify_mm 함수는 단일화를 수행하는 함수이다. 
    """단일화. 규칙 기반으로 효율성을 개선한 알고리즘(Martelli & Montanari).
    >>> unify_mm(x, 3, {})
    {x: 3}
    """
    set_eq = extend(s, x, y)
    s = set_eq.copy()
    while True:
        trans = 0
        for x, y in set_eq.items():
            if x == y:
                 # x와 y가 같은 경우 매핑이 삭제된다.
                del s[x]
            elif not is_variable(x) and is_variable(y):
                 # x가 변수가 아니고 y가 변수인 경우 y = x in s 로 다시 쓴다.
                if s.get(y, None) is None:
                    s[y] = x
                    del s[x]
                else:
                   # 변수 y에 대한 매핑이 이미 있는 경우 변수를 제거한다.
                    s[x] = vars_elimination(y, s)
            elif not is_variable(x) and not is_variable(y):
                 # x와 y가 변수가 아닌 경우, x함수와 y함수의 함수 기호가 동일한 경우이거나 x와 y의 인자 갯수가 같다면 항을 축소하고, 아니면 실패와 함께 중지한다.
                if x.op is y.op and len(x.args) == len(y.args):
                    term_reduction(x, y, s)
                    del s[x]
                else:
                    return None
            elif isinstance(y, Expr):
                #x가 변수이고, y가 함수 또는 변수인 경우
                #만약 y가 함수라면 y에서 x가 발생하는지 확인하고, 만약 x가 발생하지 않으면 실패와 함께 중지한다.
                #만약 x가 y에서 발생하면 y에 변수 제거를 적용한다.
                if occur_check(x, y, s):
                    return None
                s[x] = vars_elimination(y, s)
                if y == s.get(x):
                    trans += 1
            else:
                trans += 1
        if trans == len(set_eq):
           # 변환이 적용되지 않은 경우 성공으로 중지시킨다.
            return s
        set_eq = s.copy()
        
# 단일화가 되는지 확인하는 x, y 항을 비교하고, 두 항이 동일한지 먼저 확인한 다음, 변수인지, 함수인지 확인하면서 단일화를 수행한다.

def term_reduction(x, y, s):
    """x, y가 모두 함수이고 함수 기호가 동일한 경우 항 축소(term reduction)를 적용.
    예: x = F(x1, x2, ..., xn), y = F(x1', x2', ..., xn')
    x: y를 {x1: x1', x2: x2', ..., xn: xn'}로 대체한 새로운 매핑을 리턴.
    """
    for i in range(len(x.args)):
        if x.args[i] in s:
            s[s.get(x.args[i])] = y.args[i]
        else:
            s[x.args[i]] = y.args[i]
            
#term_reduction 함수는 x와 y가 모두 함수인 경우에 항 축소를 수행하는 함수이다. 
# 두 함수의 함수 기호가 동일한 경우와 인자의 갯수가 같은 경우에 각 인자를 대체해서 새로운 매핑을 생성한다.

def vars_elimination(x, s):
    """변수 제거를 x에 적용함.
    x가 변수이고 s에 등장하면, x에 매핑된 항을 리턴함.
    x가 함수이면 함수의 각 항에 순환적으로 적용함."""
    if not isinstance(x, Expr):
        return x
    if is_variable(x):
        return s.get(x, x)
    return Expr(x.op, *[vars_elimination(arg, s) for arg in x.args])

#vars_elimination 함수는 변수 제거를 수행하는 함수이다.
# 변수 x에 매핑된 항을 반환한다.
# 만약 x가 함수인 경우엔 함수의 각 인자에 대해 순환적으로 변수 제거를 수행한다. 

def fol_bc_ask(kb, query): #bc_ask는 backwardchain이라는 것이다.
    # 주어진 KB와 질의를 기반으로 역방향 연쇄 추론을 수행하는 함수이다.
    # KB에 질의를 해서 대입을 생성해서 결과를 반환한다.
    """역방향 연쇄(backward chaining) 알고리즘.
    kb는 FolKB 인스턴스이어야 하고, query는 기본 문장이어야 함.
    """
    return fol_bc_or(kb, query, {}) # 밑에 각 학생 별로 기숙사 입사 조건이 true인지 false 인지 비교할 때 사용된다.

def fol_bc_or(kb, goal, theta): #주어진 KB와 목표, 대입을 기반으로 역방향 연쇄 추론을 수행하는 함수이다.
    for rule in kb.fetch_rules_for_goal(goal): #fetch_rules_for_goal() 함수는 KB에서 목표와 일치하는 규칙들을 검색하여 반환한다.
        lhs, rhs = parse_definite_clause(standardize_variables(rule)) # parse_definite_clause()함수는 standardize_variables()함수를 통해 표준화된 변수들을 주어진 절을 좌변과 우변으로 분리하는 함수이다.
        for theta1 in fol_bc_and(kb, lhs, unify_mm(rhs, goal, theta)): # 아래 fol_bc_and를 통해서 목표를 달성하기 위한 대입을 theta에 대입해서 반환한다.
            yield theta1

# fol_bc_or 함수는 주어진 목표에 대해 규칙들을 검색하고, 왼쪽에 있는 조건(lhs)와 결론(rhs)를 이용해서 역방향 연쇄 알고리즘을 수행한다.
# 위에 사용했던 unify_mm 함수를 사용해서 단일화하는 작업을 재귀적으로 수행한다.

def fol_bc_and(kb, goals, theta):
    if theta is None:
        pass
    elif not goals:
        yield theta
    else:
        first, rest = goals[0], goals[1:]
        for theta1 in fol_bc_or(kb, subst(theta, first), theta):
            for theta2 in fol_bc_and(kb, rest, theta1):
                yield theta2
                
# fol_bc_and 함수는 목표와 대입을 입력으로 받아서 역방향 연쇄 알고리즘을 수행하는 함수이다.
# 목표들을 순차적으로 처리하면서 대입을 업데이트 하고, fol_bc_or 함수를 호출해서 각각에 대해 역방향 연쇄 알고리즘을 수행한다.
# 목표들이 모두 만족될 경우에 최종적으로 얻어진 대입을 반환한다.

def standardize_variables(sentence, dic=None):
    """변수 표준화: 문장의 모든 변수를 새로운 변수로 바꿈."""
    if dic is None:
        dic = {}
    if not isinstance(sentence, Expr):
        return sentence
    elif is_var_symbol(sentence.op):
        if sentence in dic:
            return dic[sentence]
        else:
            v = Expr('v_{}'.format(next(standardize_variables.counter)))
            dic[sentence] = v
            return v
    else:
        return Expr(sentence.op, *[standardize_variables(a, dic) for a in sentence.args])


standardize_variables.counter = itertools.count()

# standardize_variables 함수는 문장 내에 변수들을 새로운 변수로 표준화하는 역할을 한다.
# 함수는 문장을 입력으로 받아서 재귀적으로 탐색하면서 변수 심볼을 새로운 변수로 대체해서 변수 표준화를 수행한다.
# 이미 표준화된 변수가 있는 경우 그 변수를 반환하고, 그렇지 않으면 새로운 변수를 생성하고, 기존 변수와 대응시키는 딕셔너리에 저장한다.