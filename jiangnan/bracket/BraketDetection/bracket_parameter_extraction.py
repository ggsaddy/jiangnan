import re
def is_star_text(content=""):
    label=content.strip()
    if label==None or label=="":
        return False
    star_pattern=r"[*]+"
    return re.fullmatch(star_pattern, label)
def is_useful_text(content=""):
    label = content.strip()
    if label==None or label=="":
        return False
    if label=="B":
        return True
    pattern_b = r"(?:B)?(?P<val1>\d+([.]\d+)?)((X|x)(?P<val2>\d+([.]\d+)?))?((X|x)(?P<val3>\d+([.]\d+)?))?\s*(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_b_op = r"(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fb = r"(?:FB)?(?P<val1>\d+([.]\d+)?)((X|x)(?P<val2>\d+([.]\d+)?))?\s*(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fb_op = r"(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fl = r"(?:FL)?(?P<val1>\d+([.]\d+)?)((X|x)(?P<val2>\d+([.]\d+)?))?\s*(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fl_op=r"(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_bk = r"BK(?P<bk_code>\d+)"
    pattern_r = r"R(?P<radius>\d+([.]\d+)?)"
    pattern_digit = r"(?P<value>\d+([.]\d+)?)"
    pattern_hole=r"(?P<content>\d+(?:\.\d+)?(?:\s*[Xx]\s*\d+(?:\.\d+)?)+)"
    star_pattern=r"[*]+"
    patterns=[pattern_b, pattern_b_op, pattern_fb, pattern_fb_op, pattern_fl, pattern_fl_op, pattern_bk, pattern_r, pattern_digit,pattern_hole,star_pattern]
    flag=False
    for pattern in patterns:
        if re.fullmatch(pattern, label):
            flag=True
        if flag:
            break
    return flag

def parse_elbow_plate(label="", annotation_position="other", is_fb=False):
    """
    Parse elbow plate annotation parameters, including arm length, thickness, material, reinforcement edges, and other information.

    :param label: str, Annotation string of the elbow plate, e.g., "B150X10A", "FB120X10", "BK01", "R300"
    :param annotation_position: str, Annotation position: "top", "bottom", or "no annotation line"
    :param is_fb: bool, Force interpretation as FB type when true.
    :return: dict, A dictionary containing parameters, e.g., {"Arm Length": 150, "Thickness": 10, "Material": "A"} or None if parsing fails.
    """
    import re

    # Strip unnecessary spaces
    label = label.strip()
    if label is None or label == "":
        return None

    # Define regular expressions
    pattern_b = r"(?:B)?(?P<val1>\d+([.]\d+)?)((X|x)(?P<val2>\d+([.]\d+)?))?((X|x)(?P<val3>\d+([.]\d+)?))?\s*(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_b_op = r"(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fb = r"(?:FB)?(?P<val1>\d+([.]\d+)?)((X|x)(?P<val2>\d+([.]\d+)?))?\s*(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fb_op = r"(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fl = r"(?:FL)?(?P<val1>\d+([.]\d+)?)((X|x)(?P<val2>\d+([.]\d+)?))?\s*(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_fl_op=r"(?P<special>[~%$&]+)?\s*(?P<material>[A-Z0-9]+)?"
    pattern_bk = r"BK(?P<bk_code>\d+)"
    pattern_r = r"R(?P<radius>\d+([.]\d+)?)"
    pattern_digit = r"(?P<value>\d+([.]\d+)?)"
    pattern_hole=r"(?P<content>\d+(?:\.\d+)?(?:\s*[Xx]\s*\d+(?:\.\d+)?)+)"
    star_pattern=r"[*]+"
    if match := re.fullmatch(star_pattern, label):
        if label=="*":
            return {
                "Type": "s_star"
            } 
        else:
            return {
                "Type": "m_star"
            }          
    if label =="B":
        return {
            "Type": "B_anno"
        } 
    if match := re.fullmatch(pattern_bk, label):
        bk_code = match.group("bk_code")
        return {
            "Type": "BK",
            "Typical Section Code": bk_code,
        }
    if match := re.fullmatch(pattern_r, label):
        radius = float(match.group("radius"))
        return {
            "Type": "R",
            "Radius": radius,
            "Ref":"ref" if annotation_position=="top" or annotation_position=="bottom" else "hang"
        }
    # Check different annotation types
    if annotation_position == "top":
        # Annotations at the top prioritize parsing as B, BK, or R type
        if match := re.fullmatch(pattern_b, label):

            values = [float(v) for v in (match.group("val1"), match.group("val2"), match.group("val3")) if v]
            material = match.group("material") if match.group("material") else "AH"
            special=match.group("special") if match.group("special") else "none"
            
            arm_lengths = [v for v in values if v >= 100]
            thickness = [v for v in values if v < 100]

            if len(arm_lengths) > 2 or len(thickness) > 1:
                return None

            arm_length1 = arm_lengths[0] if len(arm_lengths) > 0 else None
            arm_length2 = arm_lengths[1] if len(arm_lengths) > 1 else None
            thickness = thickness[0] if thickness else None

            if thickness is not None and thickness > 50:
                return None

            if arm_length1 is None and arm_length2 is None and thickness is None  and material =="AH" and special=="none":
                return None


            return {
                "Type": "B",
                "Arm Length1": arm_length1,
                "Arm Length2": arm_length2,
                "Thickness": thickness,
                "Material": material,
                "Special":special
            }
        elif match := re.fullmatch(pattern_b_op, label):
            material = match.group("material") if match.group("material") else None
            special=match.group("special") if match.group("special") else "none"
            if material is not None:
                return {
                    "Type": "B",
                    "Arm Length1": None,
                    "Arm Length2": None,
                    "Thickness": None,
                    "Material": material,
                    "Special":special
                }
        elif match := re.fullmatch(pattern_bk, label):
            bk_code = match.group("bk_code")
            return {
                "Type": "BK",
                "Typical Section Code": bk_code,
            }

        elif match := re.fullmatch(pattern_r, label):
            radius = float(match.group("radius"))
            return {
                "Type": "R",
                "Radius": radius,
            }

    elif annotation_position == "bottom":
        # Annotations at the bottom prioritize parsing as FB, FL, or R type
        if is_fb:
            if match := re.fullmatch(pattern_fb, label):

                values = [float(v) for v in (match.group("val1"), match.group("val2")) if v]
                material = match.group("material") if match.group("material") else "AH"
                special=match.group("special") if match.group("special") else "none"

                section_height = [v for v in values if v >= 100]
                thickness = [v for v in values if v < 100]

                if len(section_height) > 1 or len(thickness) > 1:
                    return None

                section_height = section_height[0] if len(section_height) > 0 else None
                thickness = thickness[0] if len(thickness) > 0 else None

                if thickness is not None and thickness > 50:
                    return None

            

                return {
                    "Type": "FB",
                    "Section Height": section_height,
                    "Thickness": thickness,
                    "Material": material,
                    "Special": special
                }
            elif match := re.fullmatch(pattern_fb_op, label):
                material = match.group("material") if match.group("material") else None
                special=match.group("special") if match.group("special") else "none"
                if material is not None:
                    return {
                        "Type": "FB",
                        "Section Height": None,
                        "Thickness": None,
                        "Material": material,
                        "Special": special
                    }

        else:
            if match := re.fullmatch(pattern_fl, label):
                
                values = [float(v) for v in (match.group("val1"), match.group("val2")) if v]
                material = match.group("material") if match.group("material") else "AH"
                special=match.group("special") if match.group("special") else "none"

                section_height = [v for v in values if v >= 100]
                thickness = [v for v in values if v < 100]

                if len(section_height) > 1 or len(thickness) > 1:
                    return None

                section_height = section_height[0] if len(section_height) > 0 else None
                thickness = thickness[0] if len(thickness) > 0 else None

                if thickness is not None and thickness > 50:
                    return None


                return {
                    "Type": "FL",
                    "Section Height": section_height,
                    "Thickness": thickness,
                    "Material": material,
                    "Special":special
                }

            elif match := re.fullmatch(pattern_fb, label):
                values = [float(v) for v in (match.group("val1"), match.group("val2")) if v]
                material = match.group("material") if match.group("material") else "AH"
                special=match.group("special") if match.group("special") else "none"

                section_height = [v for v in values if v >= 100]
                thickness = [v for v in values if v < 100]

                if len(section_height) > 1 or len(thickness) > 1:
                    return None

                section_height = section_height[0] if len(section_height) > 0 else None
                thickness = thickness[0] if len(thickness) > 0 else None

                if thickness is not None and thickness > 50:
                    return None

            

                return {
                    "Type": "FB",
                    "Section Height": section_height,
                    "Thickness": thickness,
                    "Material": material,
                    "Special": special
                }
            elif match := re.fullmatch(pattern_fl_op, label):
                material = match.group("material") if match.group("material") else None
                special=match.group("special") if match.group("special") else "none"
                if material is not None:
                    return {
                        "Type": "FL",
                        "Section Height": None,
                        "Thickness": None,
                        "Material": material,
                        "Special": special
                    }

        if match := re.fullmatch(pattern_r, label):
            radius = float(match.group("radius"))
            return {
                "Type": "R",
                "Radius": radius,
            }
        elif match := re.fullmatch(pattern_bk, label):
            bk_code = match.group("bk_code")
            return {
                "Type": "BK",
                "Typical Section Code": bk_code,
            }
    elif annotation_position == "other":
        # Without an annotation line, parse as R type or a simple numerical value
        if match :=re.fullmatch(pattern_hole,label):
            content=match.group("content")
            return {
                "Type": "CornerHole",
                "Content": content
            }
            
            
        
        
        elif match := re.fullmatch(pattern_r, label):
            radius = float(match.group("radius"))
            return {
                "Type": "R",
                "Radius": radius,
            }

        elif match := re.fullmatch(pattern_digit, label):
            value = float(match.group("value"))
            return {
                "Type": "Numeric",
                "Value": value,
            }
        elif match := re.fullmatch(pattern_bk, label):
            bk_code = match.group("bk_code")
            return {
                "Type": "BK",
                "Typical Section Code": bk_code,
            }

    # If no type matches
    return None
