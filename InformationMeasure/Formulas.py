def apply_entropy_formula(prob, base):
    from scipy.stats import entropy
    return entropy(prob.flatten(), base=base)


def apply_mutual_information_formula(entropy_x, entropy_y, entropy_xy):
    return entropy_x + entropy_y - entropy_xy


def apply_conditional_mutual_information_formula(entropy_xz, entropy_yz, entropy_xyz, entropy_z):
    return entropy_xz + entropy_yz - entropy_xyz - entropy_z


def apply_conditional_entropy_formula(entropy_xz, entropy_z):
    return entropy_xz - entropy_z


def apply_total_correlation_formula(entropy_x, entropy_y, entropy_z, entropy_zyx):
    return entropy_x + entropy_y + entropy_z - entropy_zyx
