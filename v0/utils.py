import ast
def extract_name(field):
    """
    Extract the 'name' field if the value is a dict,
    otherwise return the string itself.
    """
    if isinstance(field, dict):
        return field.get("name")
    return field


def safe_parse_pass(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return {}
    return {}