level2_to_level1_code = {
    'Surface standing waters': 'C',
    'Surface running waters': 'C',
    'Dry grasslands': 'E',
    'Mesic grasslands': 'E',
    'Seasonally wet and wet grasslands': 'E',
    'Alpine and subalpine grasslands': 'E',
    'Arctic, alpine and subalpine scrub': 'F',
    'Shrub plantations': 'F',
    'Broadleaved deciduous woodland': 'G',
    'Coniferous woodland': 'G',
    'Mixed deciduous and coniferous woodland': 'G',
    'Inland cliffs, rock pavements and outcrops': 'H',
    'Arable land and market gardens': 'I',
    'Cultivated areas of gardens and parks': 'I',
    'Buildings of cities, towns and villages': 'J',
    'Low density buildings': 'J',
    'Transport networks and other constructed hard-surfaced areas': 'J',
}

level1_code_to_name = {
    'C': 'Inland surface waters',
    'E': 'Grasslands and lands dominated by forbs, mosses or lichens',
    'F': 'Heathland, scrub and tundra',
    'G': 'Woodland, forest and other wooded land',
    'H': 'Inland unvegetated or sparsely vegetated habitats',
    'I': 'Regularly or recently cultivated agricultural, horticultural and domestic habitats',
    'J': 'Constructed, industrial and other artificial habitats',
}

level2_to_level1_name = {
    level2: level1_code_to_name[code]
    for level2, code in level2_to_level1_code.items()
}
