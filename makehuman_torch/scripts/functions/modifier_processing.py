import torch
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# definitions ---

MACRO_FUNCTIONS = {
    "gender": {# human._setGenderVals
        "male": "lambda value: value", 
        "female": "lambda value: 1-value"
        }, 
    "age": {# human._setAgeVals
        "old": "lambda value: 0.0 if value < .5 else max(0.0, value * 2  - 1)",
        "baby": "lambda value: 0.0 if value > .5 else max(0.0, 1 - value * 5.333)",
        "young": "lambda value: max(0.0, (value-.1875) * 3.2) if value < .5 else 1-max(0.0, value * 2  - 1)",
        "child": "lambda value: 0.0 if value > .5 else max(0.0, min(1.0, 5.333 * value) - max(0.0, (value-.1875) * 3.2))",
        },
    "weight": { # human._setWeightVals
        "maxweight": "lambda value: max(0.0, value * 2 - 1)",
        "minweight": "lambda value: max(0.0, 1 - value * 2)",
        "averageweight": "lambda value: 1 - (max(0.0, value * 2 - 1) + max(0.0, 1 - value * 2))"
        },
    "muscle": { # human._setMuscleVals
        "maxmuscle": "lambda value: max(0.0, value * 2 - 1)",
        "minmuscle": "lambda value: max(0.0, 1 - value * 2)",
        "averagemuscle": "lambda value: 1 - (max(0.0, value * 2 - 1) + max(0.0, 1 - value * 2))"
        },
    "height": { # human._setHeightVals
        "maxheight": "lambda value: max(0.0, value * 2 - 1)",
        "minheight": "lambda value: max(0.0, 1 - value * 2)",
        "averageheight": "lambda value: 1 - max(0.0, value * 2 - 1) if max(0.0, value * 2 - 1) > max(0.0, 1 - value * 2) else 1 - max(0.0, 1 - value * 2)"
        },
    "breastsize": { # human._setBreastSizeVals
        "maxcup": "lambda value: max(0.0, value * 2 - 1)",
        "mincup": "lambda value: max(0.0, 1 - value * 2)",
        "averagecup": "lambda value: 1 - max(0.0, value * 2 - 1) if max(0.0, value * 2 - 1) > max(0.0, 1 - value * 2) else 1 - max(0.0, 1 - value * 2)"
        },
    "breastfirmness": { # human._setBreastFirmnessVals
        "maxfirmness": "lambda value: max(0.0, value * 2 - 1)",
        "minfirmness": "lambda value: max(0.0, 1 - value * 2)",
        "averagefirmness": "lambda value: 1 - max(0.0, value * 2 - 1) if max(0.0, value * 2 - 1) > max(0.0, 1 - value * 2) else 1 - max(0.0, 1 - value * 2)"
        },
    "bodyproportions": { # human._setBodyProportionsVals
        "idealproportions": "lambda value: max(0.0, value * 2 - 1)",
        "uncommonproportions": "lambda value: max(0.0, 1 - value * 2)",
        "regularproportions": "lambda value 1 - max(0.0, value * 2 - 1) if max(0.0, value * 2 - 1) > max(0.0, 1 - value * 2) else 1 - max(0.0, 1 - value * 2)"
    }
 }

ETHIC_FUNCTIONS = { #_setEthnicVals - but we are going to leverage this into a new approach they are all defined together.
        "african": "lambda value: value", # "lambda value: value[0] if value.sum() > 0 else 1/3",
        "asian": "lambda value: value", #"lambda value: value[1] if value.sum() > 0 else 1/3",
        "caucasian": "lambda value: value", #"lambda value: value[2] if value.sum() > 0 else 1/3",
}

# collection ---

def catalog_modifiers(
        human,
        modifier_classes):
    """collect information on modifiers, targets, dependences and associated 
    functions to combined them together

    Args:
        human (makehuman human): makehuman human to collect modifiers from
        modifier_classes (dictionary): needs keys: EthnicModifier, 
            MacroModifier, UniversalModifier

    Returns:
        dict: details about the set of modifiers contained in the human
    """

    if not np.all([
        np.isin(x, list(modifier_classes.keys()))
        for x in ["EthnicModifier", "MacroModifier", "UniversalModifier"] 
    ]): 
        raise ValueError(
            "modifier_classes must contain keys: EthnicModifier, MacroModifier, UniversalModifier"
        )
    catalog = []
    for modifier in human.modifiers:
        try:
            # Basic details
            modifier_info = {
                "name": modifier.fullName,
                "type": type(modifier).__name__,
                "min": modifier.getMin(),
                "max": modifier.getMax(),
                "default": modifier.getDefaultValue(),
                "is_ethnic": isinstance(modifier, modifier_classes["EthnicModifier"]),
            }

            # List impacted targets
            modifier_info["targets"] = [target[0] for target in modifier.targets]

            # Handle MacroModifier-specific logic for target functions
            if isinstance(modifier, modifier_classes["EthnicModifier"]):
                target_functions = []
                macro_variable = modifier.fullName.split('/')[-1].lower()
                inner_functions = ETHIC_FUNCTIONS[macro_variable]
                for target, dependencies in modifier.targets:
                    for dep in dependencies:
                        if dep == macro_variable:
                            inner_func = inner_functions
                        else:
                            inner_func = "" # TODO: figure out the default... (are we doing multiplicative?)
                        target_functions.append((target, dep, inner_func)) 
                        
                modifier_info["target_functions"] = target_functions

            elif isinstance(modifier, modifier_classes["MacroModifier"]):
                target_functions = []
                macro_variable = modifier.getMacroVariable()
                inner_functions = MACRO_FUNCTIONS[macro_variable]
                for target, dependencies in modifier.targets:
                    for dep in dependencies:
                        target_functions.append((target, dep, inner_functions.get(dep, ""))) # TODO: figure out the default (are we doing multiplicative?)

                modifier_info["target_functions"] = target_functions
            # Revert UniversalModifier logic to its original approach
            elif isinstance(modifier, modifier_classes["UniversalModifier"]):
                target_functions = []
                if hasattr(modifier, "left") and modifier.left:
                    target_functions.append(("left", modifier.left, [t[0] for t in modifier.targets if modifier.left in t[1]][0], "lambda value: -min(value,0.0)"))
                if hasattr(modifier, "right") and modifier.right:
                    target_functions.append(("right", modifier.right, [t[0] for t in modifier.targets if modifier.right in t[1]][0], "lambda value: max(value,0.0)"))
                if hasattr(modifier, "center") and modifier.center:
                    target_functions.append(("center", modifier.center, [t[0] for t in modifier.targets if modifier.center in t[1]][0], "lambda value: 1.0 - abs(value)"))
                modifier_info["target_functions"] = target_functions

            catalog.append(modifier_info)
        except Exception as e:
            print(f"oops {modifier.fullName}: {e}")

    return catalog

def create_target_dependency_dataframe(modifier_catalog: dict) -> pd.DataFrame:
    """
    Create a dataframe mapping targets to dependencies and associated functions.
    
    Args:
        modifier_catalog (dict): A catalog of modifiers with their targets,
            dependencies, and functions (and other details).
    
    Returns:
        pd.DataFrame: A dataframe with columns:
            - Target: The target file path.
            - Dependency: The dependency name. If associated with a 
                UniversalModifier, we add on the "left", "right" "mid" 
                structures to help with different values per target
            - Modifier Name: The name of the modifier.
            - Function: The lambda function as a string.
    """
    rows = []
    for entry in modifier_catalog:
        if "target_functions" in entry:
            if entry["type"] == "UniversalModifier":
                assert len(entry["target_functions"]) == len(entry["targets"]), (
                    f"Unexpected number of target functions for {entry['name']}"
                )
                for direction, _, target_path, function in entry["target_functions"]:
                    rows.append({
                        "Modifier Name": entry["name"],
                        "Target": target_path,
                        "Dependency": entry["name"] + "_" + direction,
                        "Function": function
                    })
            elif entry['type'] == "MacroModifier" or entry['type'] == "EthnicModifier":
                for target_path, dependency, function in entry["target_functions"]:
                    if function != "":
                        rows.append({
                            "Modifier Name": entry["name"],
                            "Target": target_path,
                            "Dependency": dependency,
                            "Function": function
                        })
    
    return pd.DataFrame(rows)[["Target", "Dependency", "Modifier Name", "Function"]]

def create_modifier_info_dataframe(modifier_catalog):
    """
    Create a dataframe summarizing modifier information
    
    Args:
        modifier_catalog (dict): A catalog of modifiers with their targets,
            dependencies, and functions (and other details).
    
    Returns:
        pd.DataFrame: A dataframe with columns:
            - Modifier Name: name of modifier
            - Min: minimum value modifier can take 
            - Max: maximum value modifier can take
            - Default: default value of the modifier
            - Comment: special comments about the modifier
    """
    rows = []
    for entry in modifier_catalog:
        if "target_functions" in entry:
            if entry["type"] == "UniversalModifier":
                rows.append({
                    "Modifier Name": entry["name"],
                    "Min": entry["min"],
                    "Max": entry["max"],
                    "Default": entry["default"],
                    "Comment": ""
                })
            elif entry['type'] == "MacroModifier" or entry['type'] == "EthnicModifier":
                if entry["type"] == "EthnicModifier":
                    comment = "Ethnic modifier operates in a unit circle for dependencies."
                else:
                    comment = ""

                rows.append({
                    "Modifier Name": entry["name"],
                    "Min": entry["min"],
                    "Max": entry["max"],
                    "Default": entry["default"],
                    "Comment": comment
                })
    
    return pd.DataFrame(rows)

# To data structure ---

def target_to_dependencies_mapping(
        target_key: Dict[str, int],
        dependency_key: Dict[str, int],
        mapping: List[Tuple[str, str]],
        device=torch.device('cpu')
    ) -> torch.tensor:
    """Create a sparse binary mapping tensor from targets to dependencies.

    Args:
        target_key (dict of str): list of target names corresponding to
            the indices in the target tensor.
        dependency_key (dict of str): list of dependency names corresponding to
            the indices in the dependency tensor.
        mapping (list): mapping from target names to lists of dependency
            names they depend on.

    Returns:
        torch.tensor: sparse binary mapping of target to set of
            dependencies. (t, d)
    """

    rows, cols = [], []
    for target, dep in mapping:
        if target not in target_key:
            raise ValueError(
                f"Target {target} not found in target_key."
            )
        t_idx = target_key[target]

        if dep not in dependency_key:
            raise ValueError(
                f"Dependency {dep} not found in dependency_key."
            )
        d_idx = dependency_key[dep]
        rows.append(t_idx)
        cols.append(d_idx)

    all_indicies = torch.tensor([rows, cols], device=device)
    values = torch.ones(len(rows), device=device)
    mapping_tensor = torch.sparse_coo_tensor(
        all_indicies, 
        values, 
        (len(target_key), len(dependency_key)),
        device=device
    )

    return mapping_tensor

# def calc_modifiers_to_dependencies(
#         modifiers: torch.tensor, 
#         modifier_key: Dict[str, int], 
#         dependency_key: Dict[str, int], 
#         mapping: List[Tuple[str, str, str]]):
#     """Convert of modifier values to a vector of dependencies.
    
#     Args:
#         modifiers (torch.tensor): scalar values for each modifier m, 
#             representing its influence on the body mesh. (m,) Should be full
#             of values (including default values).
#         modifier_key (dict of str): list of modifier names corresponding to
#             the indices in the modifier tensor.
#         dependency_key (list of str): list of dependency names to output.
#         mapping (dict): mapping from modifier names to lists of dependency
#             names they influence and the lambda function associated with them.
#             The lambda function should use torch functionality only.

#     Returns:
#         torch.tensor: scalar values for each dependency d as a function of 
#             the modifier values. In order of the dependency_key
#     """
#     dependencies = torch.ones(len(dependency_key), device=modifiers.device)

#     for dep, mod, function in mapping:
#         if mod not in modifier_key:
#             raise ValueError(
#                 f"Modifier {mod} not found in modifier_key."
#             )
#         if dep not in dependency_key:
#             raise ValueError(
#                 f"Dependency {dep} not found in dependency_key."
#             )

#         i_d, i_m = dependency_key[dep], modifier_key[mod]

#         dependencies[i_d] = eval(function)(modifiers[i_m])

#     return dependencies, dependency_key

# def calc_dependencies_to_targets(
#         dependencies: torch.tensor, 
#         dependencies_key: Dict[str, int],
#         mapping: torch.tensor,
#         mapping_keys: Tuple[Dict, Dict]):
#     """Convert of dependency values to a vector of targets.
    
#     Args:
#         dependencies (torch.tensor): scalar values for each dependency d. (d,)
#         mapping_tensor (torch.tensor): sparse binary mapping of target to set of 
#             dependencies. (t, d)
#         mapping_keys (tuple): (target_key, dependency_key) dictionaries.

#     Returns:
#         torch.tensor: scalar values for each target t as a function of 
#             the dependency values. In order of the target_key (t,)
#     """
#     log_dependencies = torch.log(dependencies)
#     targets_log = torch.sparse.mm(mapping, log_dependencies) # assumed to handle -inf well...
#     targets = torch.exp(targets_log)

#     return targets