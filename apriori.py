#
# Apriori
#

from itertools import combinations
from pandas import DataFrame

# arguments:
#   df: dataframe (DataFrame in Pandas, each element must be True or False)
#   min_supp: minimum support (optional, default is 0.1)
#   min_conf: minimum confidence (optional, default is 0.8)
#   verbose: show verbose output (optional, default is False)
# return value:
#   dataframe of rules
def apriori(df, min_supp = 0.1, min_conf = 0.8, verbose=False):
    # minimum support (=ratio) is converted to threshold for actual count
    min_supp_count = len(df) * min_supp

    # generate bool index table for faster calculation(?)
    bool_indexes = { i:(df[i]==True) for i in df.columns }

    # initialize sets of column (variable) names
    # (list of sets which contain only one column name)
    var_set = [ set([i]) for i in df.columns ]

    num = 1
    rules = []
    var_all = []

    # main loop: continues while var_set is not empty
    while len(var_set) > 0:
        # calculate count (i.e., support) of each element and keep it if its count is larger than the threshold
        var_surv = [ v for v in var_set if count_basket(bool_indexes, v) >= min_supp_count ]
        var_all += var_surv

        # calculate confidence for each of survived set and extract as rule string
        r = []
        for v in var_surv:
            r += calc_conf(bool_indexes, len(df), min_conf, v)
        rules += r
        if verbose:
            print("Length %d: total %d sets, %d survived, %d rules." % (num, len(var_set), len(var_surv), len(r)))

        # extend survived sets by adding one variable to each set
        var_set = extend_set(var_surv)
        num += 1

    if verbose:
        print("Frequent Item Sets:")
        print(var_all)

    # if you change function show_rule below to use format_rule, do not forget to comment these lines
    rules = DataFrame(rules)
    rules.columns = ["LHS", "RHS", "Count", "Support", "Confidence", "Lift"]        
        
    return rules

# arguments:
#   bi: dictionary of bool indexes
#   item_set: set of items to be counted
# return value:
#   conut of baskets which contain the set of items
def count_basket(bi, item_set):
    z = []
    for i in item_set:
        if len(z) == 0:
            z = bi[i]
        else:
            z = z & bi[i]
    return sum(z)
    
# arguments:
#   freq_set: list of sets which are Frequent Item Sets
# return value:
#   list of sets, each of which contains one more item
def extend_set(freq_list):
    # create set of individual items appeared in freq_list
    items = set() # empty set
    for i in freq_list:
        items = items.union(i)
    
    # add an item in items to each of given sets unless already contained
    new_set = []
    for s in freq_list:
        for i in items:
            if i in s:
                continue
            t = s | set([i])
            if t not in new_set:
                new_set.append(t)
                
    return new_set

# arguments:
#   bi: dictionary of bool indexes
#   df_len: number of rows in the dataframe (=total number of baskets)
#   threshold: minimum confidence threshold
#   item_set: set of items to be considered
# return value:
#   list of rules in string, each of which contains support, confidence and lift values
def calc_conf(bi, df_len, threshold, item_set):
    items = list(item_set)
  
    # count baskets which contain all items in item_set to calculate support
    t = count_basket(bi, item_set)
    supp = t / df_len
    
    # initialize list of rules
    if supp >= threshold:
        # add a rule whose LHS is empty, if confidence is equal or larger than the threshold
        # when LHS is empty, suport=confidence, thus lift=1.0
        rules = [ show_rule(set(), item_set, t, supp, supp, 1.0 ) ]
    else:
        rules = []

    # prepare list of LHS items (=list of combinations of items in item_set)
    items_c = [ ]
    for i in range(1, len(items)):
        items_c += [ set(c) for c in combinations(items, r=i) ]

    # calculate confidence for each
    for c in items_c:
        # RHS items
        s = item_set - c
        # calcualte confidence, and skip this if smaller than the threshold
        conf = t / count_basket(bi, c)
        if conf < threshold:
            continue
        # calculate lift and generate rule string
        lift = conf / ( count_basket(bi, s) / df_len )
        r = show_rule(c, s, t, supp, conf, lift)
        rules.append(r)
        
    return(rules)

# arguments:
#   lh: set of items in left hand side
#   rh: set of items in right hand side
#   n: total count of lh | rh (union)
#   s: support value
#   c: confidence value
#   l: lift value
# return value:
#   rule string (format_rule) or list of rule (list_rule)
def show_rule(lh, rh, n, s, c, l):
    # if you want to have rules in string, instead of list, change here
    # return format_rule(lh, rh, n, s, c, l)
    return list_rule(lh, rh, n, s, c, l)

def format_rule(lh, rh, n, s, c, l):
    if len(lh) == 0:
        lhs = "(None)"
    else:
        lhs = ",".join(list(lh))

    if len(rh) == 0:
        rhs = "(None)"
    else:
        rhs = ",".join(list(rh))
    
    return "%s -> %s : count=%d supp=%.3f conf=%.3f lift=%.3f" % (lhs, rhs, n, s, c, l)

def list_rule(lh, rh, n, s, c, l):
    if len(lh) == 0:
        lhs = ""
    else:
        lhs = ",".join(list(lh))

    if len(rh) == 0:
        rhs = ""
    else:
        rhs = ",".join(list(rh))
    
    return [lhs, rhs, n, s, c, l]
