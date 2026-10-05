"""
utils/machine_registry_additions.py

New machine profiles: cultivator, sprayer, drip_irrigation.

HOW TO USE
──────────
In your existing utils/machine_registry.py, at the bottom of the file
where all other profiles are registered, call:

    from utils.machine_registry_additions import register_new_machines
    register_new_machines(registry)

where `registry` is whatever dict/object _REGISTRY refers to in your code.

If machine_registry.py uses a list of MachineProfile objects, append with:

    from utils.machine_registry_additions import NEW_MACHINE_PROFILES
    _PROFILES.extend(NEW_MACHINE_PROFILES)  # or however your registry is built

The data below is self-contained — no imports from machine_registry.py —
so it works regardless of the exact registry implementation.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# These three dicts mirror the structure of every existing machine profile.
# Field names match what diagnosis_service.py, repair_agent.py, and
# verification_service.py read via get_profile_or_default(), etc.
# ─────────────────────────────────────────────────────────────────────────────

CULTIVATOR_PROFILE = {
    "machine_id":      "cultivator",
    "label_en":        "Cultivator",
    "label_hi":        "कल्टीवेटर",
    "category":        "tractor_implement",
    "is_tractor_attachment": True,
    "is_electric":     False,

    "farmer_intro_en": (
        "A cultivator is an implement attached to the back of your tractor. "
        "It has rows of metal tines or shovels that dig into the soil to break clods, "
        "remove weeds, and aerate the ground between crop rows."
    ),
    "farmer_intro_hi": (
        "कल्टीवेटर ट्रैक्टर के पीछे लगने वाला औज़ार है। "
        "इसमें धातु के दाँते होते हैं जो मिट्टी तोड़ते हैं, खरपतवार निकालते हैं "
        "और फसलों के बीच मिट्टी को हवादार बनाते हैं।"
    ),

    # ── Area zones ────────────────────────────────────────────────────────────
    # Cultivator has no engine — zones are purely implement-side.
    "area_zones": [
        {
            "id":                       "implement_frame",
            "label_en":                 "Main Frame & Toolbar",
            "label_hi":                 "मुख्य फ्रेम और टूलबार",
            "farmer_description_en":    "The heavy steel frame and horizontal bar the tines bolt onto",
            "farmer_description_hi":    "वह मोटा लोहे का ढाँचा जिस पर दाँते लगे हैं",
        },
        {
            "id":                       "tines_and_shovels",
            "label_en":                 "Tines, Shovels & Sweeps",
            "label_hi":                 "दाँते और फालियाँ",
            "farmer_description_en":    "The curved metal spikes or flat blades that go into the soil",
            "farmer_description_hi":    "मिट्टी में घुसने वाले मुड़े हुए दाँते या चपटी फालियाँ",
        },
        {
            "id":                       "depth_adjustment",
            "label_en":                 "Depth Wheel & Adjustment",
            "label_hi":                 "गहराई पहिया और समायोजन",
            "farmer_description_en":    "The small wheel and bolt/pin that sets how deep the tines go",
            "farmer_description_hi":    "छोटा पहिया और बोल्ट जो दाँतों की गहराई तय करता है",
        },
        {
            "id":                       "hitch_linkage",
            "label_en":                 "3-Point Hitch Linkage",
            "label_hi":                 "तीन बिंदु हिच लिंकेज",
            "farmer_description_en":    "The top-link and two lower arms that connect to the tractor's hydraulic arms",
            "farmer_description_hi":    "ट्रैक्टर के हाइड्रोलिक आर्म से जुड़ने वाले तीन जोड़",
        },
    ],

    # ── Parts ─────────────────────────────────────────────────────────────────
    "parts": [
        {
            "id": "main_frame",
            "area_zone": "implement_frame",
            "label_en": "Main Frame",
            "label_hi": "मुख्य फ्रेम",
            "farmer_description_en": "The thick rectangular steel bar running across the width of the cultivator",
            "ar_model": "cultivator_frame.obj",
        },
        {
            "id": "tine",
            "area_zone": "tines_and_shovels",
            "label_en": "Tine / Shank",
            "label_hi": "दाँता / शैंक",
            "farmer_description_en": "The curved spring-steel or rigid metal spike that enters the soil",
            "ar_model": "cultivator_tine.obj",
        },
        {
            "id": "shovel_point",
            "area_zone": "tines_and_shovels",
            "label_en": "Shovel Point / Sweep",
            "label_hi": "फाली / स्वीप",
            "farmer_description_en": "The replaceable flat blade bolted to the bottom of each tine",
            "ar_model": "cultivator_shovel.obj",
        },
        {
            "id": "tine_bolt",
            "area_zone": "tines_and_shovels",
            "label_en": "Tine Clamp Bolt",
            "label_hi": "दाँता क्लैंप बोल्ट",
            "farmer_description_en": "The U-bolt or clamp that secures the tine to the toolbar",
            "ar_model": "cultivator_bolt.obj",
        },
        {
            "id": "depth_wheel",
            "area_zone": "depth_adjustment",
            "label_en": "Depth Control Wheel",
            "label_hi": "गहराई नियंत्रण पहिया",
            "farmer_description_en": "Small rubber wheel at the side that rolls on the ground surface",
            "ar_model": "depth_wheel.obj",
        },
        {
            "id": "depth_pin",
            "area_zone": "depth_adjustment",
            "label_en": "Depth Setting Pin",
            "label_hi": "गहराई पिन",
            "farmer_description_en": "A metal pin or bolt that locks the depth wheel at the correct height",
            "ar_model": "depth_pin.obj",
        },
        {
            "id": "top_link",
            "area_zone": "hitch_linkage",
            "label_en": "Top Link",
            "label_hi": "ऊपरी लिंक",
            "farmer_description_en": "The adjustable rod at the top connecting cultivator to tractor hitch",
            "ar_model": "top_link.obj",
        },
        {
            "id": "lower_link_pin",
            "area_zone": "hitch_linkage",
            "label_en": "Lower Link Pin",
            "label_hi": "निचला लिंक पिन",
            "farmer_description_en": "The thick pin that locks the lower hydraulic arms to the implement",
            "ar_model": "link_pin.obj",
        },
    ],

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    "diagnostic_context": (
        "Cultivators are tractor-mounted implements — they have no engine of their own. "
        "All problems are mechanical: bent/broken tines, worn shovel points, loose tine bolts, "
        "incorrect depth setting, or faulty 3-point hitch linkage. "
        "Start by checking the tines and shovel points (most common wear items), "
        "then depth wheel, then hitch pins."
    ),
    "compact_diagnostic_hint": (
        "tines_worn→shovel_point | loose_bolt→tine_bolt | uneven_depth→depth_wheel+depth_pin | "
        "implement_tilts→top_link | hitch_won't_raise→lower_link_pin+tractor_hydraulic"
    ),
    "compact_parts_list": (
        "main_frame | tine | shovel_point | tine_bolt | depth_wheel | depth_pin | "
        "top_link | lower_link_pin"
    ),
    "compact_safety_keywords": (
        "PTO_off_before_touching | lower_implement_to_ground_before_adjusting | "
        "sharp_tines_cut_hands | never_stand_behind_while_tractor_running"
    ),

    # ── Parts that escalate if damaged ────────────────────────────────────────
    "critical_parts": ["main_frame", "lower_link_pin", "top_link"],
    "fuel_system_parts": [],   # no engine

    # ── Safety ────────────────────────────────────────────────────────────────
    "base_safety_warnings_en": [
        "Lower the cultivator to the ground and switch off the tractor before touching any tine.",
        "Tine points and shovel edges are extremely sharp — wear thick gloves.",
        "Never stand or walk behind the cultivator while the tractor engine is running.",
        "Disengage PTO and stop tractor before adjusting depth wheel or removing pins.",
    ],
    "base_safety_warnings_hi": [
        "किसी भी दाँते को छूने से पहले कल्टीवेटर जमीन पर उतारें और ट्रैक्टर बंद करें।",
        "दाँते और फाली की धार बहुत तेज़ होती है — मोटे दस्ताने पहनें।",
        "ट्रैक्टर चलते समय कल्टीवेटर के पीछे कभी न खड़े हों।",
        "गहराई पहिया बदलने या पिन निकालने से पहले PTO बंद करें और ट्रैक्टर रोकें।",
    ],
}


SPRAYER_PROFILE = {
    "machine_id":      "sprayer",
    "label_en":        "Crop Sprayer",
    "label_hi":        "स्प्रेयर / दवाई मशीन",
    "category":        "crop_protection",
    "is_tractor_attachment": False,
    "is_electric":     False,   # most knapsack/engine sprayers are petrol or manual

    "farmer_intro_en": (
        "A sprayer pumps pesticide, herbicide, or fertiliser solution through a nozzle "
        "as a fine mist onto crops. It can be a backpack (knapsack) type carried by the farmer, "
        "a motorised engine sprayer, or a boom sprayer mounted on a tractor."
    ),
    "farmer_intro_hi": (
        "स्प्रेयर कीटनाशक, खरपतवारनाशी या खाद का घोल फसल पर महीन फुहार के रूप में छिड़कता है। "
        "यह पीठ पर लादने वाला (नैपसैक), मोटर वाला इंजन स्प्रेयर, "
        "या ट्रैक्टर पर लगा बूम स्प्रेयर हो सकता है।"
    ),

    "area_zones": [
        {
            "id":                       "pressure_tank",
            "label_en":                 "Pressure Tank & Cap",
            "label_hi":                 "प्रेशर टैंक और ढक्कन",
            "farmer_description_en":    "The main container holding the chemical solution",
            "farmer_description_hi":    "दवाई का घोल रखने वाला मुख्य डिब्बा",
        },
        {
            "id":                       "pump_unit",
            "label_en":                 "Pump (piston / diaphragm)",
            "label_hi":                 "पंप (पिस्टन / डायाफ्राम)",
            "farmer_description_en":    "The mechanism that builds pressure to push liquid to the nozzle",
            "farmer_description_hi":    "वह यंत्र जो दवाई को दबाव से नोज़ल तक भेजता है",
        },
        {
            "id":                       "hose_and_lance",
            "label_en":                 "Hose, Lance & Nozzle",
            "label_hi":                 "होस, लांस और नोज़ल",
            "farmer_description_en":    "The rubber pipe, metal or plastic wand, and spray tip at the end",
            "farmer_description_hi":    "रबड़ की नली, छड़ी और नोक जहाँ से दवाई निकलती है",
        },
        {
            "id":                       "engine_unit",
            "label_en":                 "Engine / Motor Unit",
            "label_hi":                 "इंजन / मोटर",
            "farmer_description_en":    "The petrol engine or electric motor that powers the pump (motorised sprayers only)",
            "farmer_description_hi":    "पेट्रोल इंजन या मोटर जो पंप चलाती है (केवल मोटर वाले स्प्रेयर में)",
        },
    ],

    "parts": [
        {
            "id": "tank_body",
            "area_zone": "pressure_tank",
            "label_en": "Tank Body",
            "label_hi": "टैंक बॉडी",
            "farmer_description_en": "The plastic or metal container that holds the spray solution",
            "ar_model": "sprayer_tank.obj",
        },
        {
            "id": "tank_cap_seal",
            "area_zone": "pressure_tank",
            "label_en": "Tank Cap & Rubber Seal",
            "label_hi": "टैंक ढक्कन और रबड़ सील",
            "farmer_description_en": "The screw-on lid and the rubber ring inside it that prevents leaks",
            "ar_model": "tank_cap.obj",
        },
        {
            "id": "pressure_relief_valve",
            "area_zone": "pressure_tank",
            "label_en": "Pressure Relief Valve",
            "label_hi": "प्रेशर रिलीफ वाल्व",
            "farmer_description_en": "A small brass or plastic valve that releases pressure if it gets too high",
            "ar_model": "relief_valve.obj",
        },
        {
            "id": "pump_piston_seal",
            "area_zone": "pump_unit",
            "label_en": "Pump Piston / Diaphragm Seal",
            "label_hi": "पंप पिस्टन / डायाफ्राम सील",
            "farmer_description_en": "The rubber seal inside the pump that wears out and causes low pressure",
            "ar_model": "pump_seal.obj",
        },
        {
            "id": "filter_strainer",
            "area_zone": "pump_unit",
            "label_en": "Filter / Strainer",
            "label_hi": "फिल्टर / जाली",
            "farmer_description_en": "A small mesh screen that catches dirt before it blocks the nozzle",
            "ar_model": "sprayer_filter.obj",
        },
        {
            "id": "delivery_hose",
            "area_zone": "hose_and_lance",
            "label_en": "Delivery Hose",
            "label_hi": "डिलीवरी होस",
            "farmer_description_en": "The flexible rubber pipe running from the tank to the lance",
            "ar_model": "delivery_hose.obj",
        },
        {
            "id": "nozzle_tip",
            "area_zone": "hose_and_lance",
            "label_en": "Nozzle Tip",
            "label_hi": "नोज़ल टिप",
            "farmer_description_en": "The small plastic or brass tip at the end of the wand where spray comes out",
            "ar_model": "nozzle_tip.obj",
        },
        {
            "id": "trigger_valve",
            "area_zone": "hose_and_lance",
            "label_en": "Trigger / Shut-off Valve",
            "label_hi": "ट्रिगर / शट-ऑफ वाल्व",
            "farmer_description_en": "The lever or trigger you squeeze to start and stop the spray",
            "ar_model": "trigger_valve.obj",
        },
        {
            "id": "engine_air_filter",
            "area_zone": "engine_unit",
            "label_en": "Engine Air Filter",
            "label_hi": "इंजन एयर फिल्टर",
            "farmer_description_en": "A foam or paper filter that stops dust entering the engine (motorised only)",
            "ar_model": "air_filter.obj",
        },
        {
            "id": "engine_fuel_cap",
            "area_zone": "engine_unit",
            "label_en": "Fuel Tank Cap",
            "label_hi": "ईंधन ढक्कन",
            "farmer_description_en": "The cap on the petrol tank of the engine (motorised only)",
            "ar_model": "fuel_cap.obj",
        },
    ],

    "diagnostic_context": (
        "Sprayer problems fall into two groups: "
        "(1) No or low spray — clogged nozzle tip (most common), blocked filter strainer, "
        "worn pump seal, or leaking tank cap seal. "
        "(2) Engine won't start (motorised) — check fuel level, dirty air filter, or spark plug. "
        "Always depressurise the tank before opening any part. "
        "Chemical exposure is a serious hazard — farmer must wear gloves and mask."
    ),
    "compact_diagnostic_hint": (
        "no_spray→nozzle_tip(clogged)+filter_strainer | low_pressure→pump_piston_seal | "
        "drip_leak→tank_cap_seal+delivery_hose | engine_no_start→fuel+air_filter | "
        "spray_uneven→nozzle_tip(worn)"
    ),
    "compact_parts_list": (
        "tank_body | tank_cap_seal | pressure_relief_valve | pump_piston_seal | "
        "filter_strainer | delivery_hose | nozzle_tip | trigger_valve | "
        "engine_air_filter | engine_fuel_cap"
    ),
    "compact_safety_keywords": (
        "depressurise_before_opening | gloves_and_mask_always | "
        "never_spray_upwind | rinse_tank_after_use | keep_children_away"
    ),

    "critical_parts": ["pressure_relief_valve", "trigger_valve"],
    "fuel_system_parts": ["engine_fuel_cap"],

    "base_safety_warnings_en": [
        "Always release tank pressure before opening the cap or any fitting.",
        "Wear rubber gloves and a face mask — pesticide chemicals are toxic.",
        "Never spray into the wind — it will blow chemical back onto your face.",
        "Rinse the tank and nozzle with clean water after every use.",
        "Keep children and animals away from the spraying area.",
    ],
    "base_safety_warnings_hi": [
        "ढक्कन या कोई भी फिटिंग खोलने से पहले टैंक का दबाव हमेशा छोड़ें।",
        "रबड़ के दस्ताने और मास्क ज़रूर पहनें — कीटनाशक ज़हरीला होता है।",
        "हवा की दिशा में कभी न छिड़कें — दवाई आपके मुँह पर आ सकती है।",
        "हर उपयोग के बाद टैंक और नोज़ल को साफ पानी से धोएं।",
        "छिड़काव वाले क्षेत्र से बच्चों और जानवरों को दूर रखें।",
    ],
}


DRIP_IRRIGATION_PROFILE = {
    "machine_id":      "drip_irrigation",
    "label_en":        "Drip / Sprinkler Irrigation System",
    "label_hi":        "ड्रिप / स्प्रिंकलर सिंचाई प्रणाली",
    "category":        "irrigation",
    "is_tractor_attachment": False,
    "is_electric":     False,   # pump is separate; system itself is passive pipes

    "farmer_intro_en": (
        "A drip or sprinkler irrigation system delivers water directly to crop roots "
        "through a network of pipes, emitters, or sprinkler heads. "
        "It saves water and fertiliser compared to flood irrigation. "
        "Problems are usually blocked emitters/nozzles, leaking joints, or filter blockages."
    ),
    "farmer_intro_hi": (
        "ड्रिप या स्प्रिंकलर सिंचाई प्रणाली पाइपों के जाल के ज़रिए "
        "सीधे फसल की जड़ों तक पानी पहुँचाती है। "
        "यह बाढ़ सिंचाई की तुलना में पानी और खाद बचाती है। "
        "अधिकतर समस्याएं ड्रिपर/नोज़ल का बंद होना, जोड़ का लीक होना, "
        "या फिल्टर का जाम होना होती हैं।"
    ),

    "area_zones": [
        {
            "id":                       "filter_unit",
            "label_en":                 "Filter Unit & Pressure Regulator",
            "label_hi":                 "फिल्टर यूनिट और प्रेशर रेगुलेटर",
            "farmer_description_en":    "The filter housing and pressure gauge mounted near the water source",
            "farmer_description_hi":    "पानी के स्रोत के पास लगा फिल्टर और प्रेशर गेज",
        },
        {
            "id":                       "mainline_pipes",
            "label_en":                 "Mainline & Sub-main Pipes",
            "label_hi":                 "मेन पाइप और सब-मेन पाइप",
            "farmer_description_en":    "The thick black pipes running along the field edge",
            "farmer_description_hi":    "खेत के किनारे बिछी मोटी काली पाइप",
        },
        {
            "id":                       "lateral_pipes",
            "label_en":                 "Lateral Pipes & Drippers",
            "label_hi":                 "लेटरल पाइप और ड्रिपर",
            "farmer_description_en":    "Thin black pipes running along crop rows with small drippers or emitters",
            "farmer_description_hi":    "फसल की कतारों में बिछी पतली काली पाइप और उनमें लगे ड्रिपर",
        },
        {
            "id":                       "sprinkler_heads",
            "label_en":                 "Sprinkler Heads & Risers",
            "label_hi":                 "स्प्रिंकलर हेड और राइज़र",
            "farmer_description_en":    "The rotating or fixed spray heads mounted on vertical pipes above the crop",
            "farmer_description_hi":    "ऊर्ध्वाधर पाइपों पर लगे घूमने वाले या स्थिर फुहारे",
        },
        {
            "id":                       "control_valves",
            "label_en":                 "Control Valves & Manifold",
            "label_hi":                 "कंट्रोल वाल्व और मैनिफोल्ड",
            "farmer_description_en":    "The on/off and zone valves that control which section gets water",
            "farmer_description_hi":    "वाल्व जो तय करते हैं कि किस हिस्से में पानी जाएगा",
        },
    ],

    "parts": [
        {
            "id": "screen_filter",
            "area_zone": "filter_unit",
            "label_en": "Screen Filter",
            "label_hi": "स्क्रीन फिल्टर",
            "farmer_description_en": "A mesh cylinder inside the filter housing that traps sand and debris",
            "ar_model": "screen_filter.obj",
        },
        {
            "id": "pressure_gauge",
            "area_zone": "filter_unit",
            "label_en": "Pressure Gauge",
            "label_hi": "प्रेशर गेज",
            "farmer_description_en": "A round dial showing water pressure — should read 1–2 bar for drip",
            "ar_model": "pressure_gauge.obj",
        },
        {
            "id": "pressure_regulator",
            "area_zone": "filter_unit",
            "label_en": "Pressure Regulator",
            "label_hi": "प्रेशर रेगुलेटर",
            "farmer_description_en": "A brass or plastic device that limits pressure to protect drippers",
            "ar_model": "pressure_regulator.obj",
        },
        {
            "id": "mainline_joint",
            "area_zone": "mainline_pipes",
            "label_en": "Mainline Joint / Coupling",
            "label_hi": "मेन पाइप जोड़",
            "farmer_description_en": "The plastic or rubber connector joining two sections of thick mainline pipe",
            "ar_model": "pipe_joint.obj",
        },
        {
            "id": "ball_valve",
            "area_zone": "control_valves",
            "label_en": "Ball Valve",
            "label_hi": "बॉल वाल्व",
            "farmer_description_en": "A round-handled valve that turns 90° to open or close water flow",
            "ar_model": "ball_valve.obj",
        },
        {
            "id": "lateral_pipe",
            "area_zone": "lateral_pipes",
            "label_en": "Lateral Pipe",
            "label_hi": "लेटरल पाइप",
            "farmer_description_en": "The thin black pipe (12–16 mm) laid along each crop row",
            "ar_model": "lateral_pipe.obj",
        },
        {
            "id": "dripper_emitter",
            "area_zone": "lateral_pipes",
            "label_en": "Dripper / Emitter",
            "label_hi": "ड्रिपर / एमिटर",
            "farmer_description_en": "A small black or brown button-shaped device that releases water drop by drop",
            "ar_model": "dripper.obj",
        },
        {
            "id": "grommet_takeoff",
            "area_zone": "lateral_pipes",
            "label_en": "Grommet / Takeoff Fitting",
            "label_hi": "ग्रॉमेट / टेकऑफ फिटिंग",
            "farmer_description_en": "The small rubber seal and plastic fitting where the lateral pipe connects to the mainline",
            "ar_model": "grommet.obj",
        },
        {
            "id": "sprinkler_head",
            "area_zone": "sprinkler_heads",
            "label_en": "Sprinkler Head",
            "label_hi": "स्प्रिंकलर हेड",
            "farmer_description_en": "The rotating or fixed head that throws water in a circle pattern",
            "ar_model": "sprinkler_head.obj",
        },
        {
            "id": "riser_pipe",
            "area_zone": "sprinkler_heads",
            "label_en": "Riser Pipe",
            "label_hi": "राइज़र पाइप",
            "farmer_description_en": "The short vertical pipe that lifts the sprinkler head above the crop",
            "ar_model": "riser_pipe.obj",
        },
    ],

    "diagnostic_context": (
        "Drip/sprinkler systems have no engine. All faults are hydraulic or mechanical: "
        "(1) No water from a dripper — check if dripper is clogged (most common — calcium/algae deposits); "
        "flush or replace dripper. "
        "(2) Low pressure across whole system — screen filter is blocked; clean filter mesh. "
        "(3) Leaking joint — grommet seal failed or mainline joint cracked. "
        "(4) Sprinkler not rotating — debris inside sprinkler head; remove and clean. "
        "Always check pressure gauge first — normal operating pressure is 1–2 bar for drip, "
        "2–4 bar for sprinklers."
    ),
    "compact_diagnostic_hint": (
        "no_drip→dripper_emitter(clogged) | low_whole_system→screen_filter | "
        "leak_at_row→grommet_takeoff | sprinkler_not_spinning→sprinkler_head(debris) | "
        "pressure_zero→ball_valve(closed)+mainline_joint(burst)"
    ),
    "compact_parts_list": (
        "screen_filter | pressure_gauge | pressure_regulator | mainline_joint | "
        "ball_valve | lateral_pipe | dripper_emitter | grommet_takeoff | "
        "sprinkler_head | riser_pipe"
    ),
    "compact_safety_keywords": (
        "close_pump_before_opening_any_joint | flush_filter_monthly | "
        "never_walk_on_lateral_pipes | winterise_before_frost"
    ),

    "critical_parts": ["screen_filter", "pressure_regulator", "ball_valve"],
    "fuel_system_parts": [],

    "base_safety_warnings_en": [
        "Turn off the pump and release line pressure before opening any fitting or filter.",
        "Flush the filter screen at least once a month to prevent pressure loss.",
        "Never walk on lateral pipes — they crack and are expensive to replace.",
        "Check for emitter blockages after fertigation — fertiliser salts clog drippers quickly.",
    ],
    "base_safety_warnings_hi": [
        "कोई भी फिटिंग या फिल्टर खोलने से पहले पंप बंद करें और पाइप का दबाव छोड़ें।",
        "दबाव कम होने से बचाने के लिए फिल्टर स्क्रीन महीने में कम से कम एक बार साफ करें।",
        "लेटरल पाइपों पर कभी न चलें — वे टूट जाती हैं और महंगी होती हैं।",
        "फर्टिगेशन के बाद ड्रिपर की जाँच करें — खाद के नमक ड्रिपर जल्दी बंद करते हैं।",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Alias map — all the ways a farmer or Flutter might name these machines.
# Add these entries to the existing _ALIASES dict in machine_registry.py.
# ─────────────────────────────────────────────────────────────────────────────

NEW_MACHINE_ALIASES: dict[str, str] = {
    # Cultivator
    "cultivator":           "cultivator",
    "kultivator":           "cultivator",
    "kulti":                "cultivator",
    "tine cultivator":      "cultivator",
    "soil cultivator":      "cultivator",
    "inter cultivator":     "cultivator",
    "weeder":               "cultivator",
    "dantali":              "cultivator",

    # Sprayer
    "sprayer":              "sprayer",
    "crop sprayer":         "sprayer",
    "pesticide machine":    "sprayer",
    "pesticide sprayer":    "sprayer",
    "knapsack sprayer":     "sprayer",
    "power sprayer":        "sprayer",
    "boom sprayer":         "sprayer",
    "mist blower":          "sprayer",
    "fog machine":          "sprayer",
    "dawa machine":         "sprayer",
    "spray machine":        "sprayer",

    # Drip / sprinkler / irrigation
    "drip irrigation":      "drip_irrigation",
    "drip system":          "drip_irrigation",
    "drip":                 "drip_irrigation",
    "micro irrigation":     "drip_irrigation",
    "micro drip":           "drip_irrigation",
    "sprinkler":            "drip_irrigation",
    "sprinkler system":     "drip_irrigation",
    "rain gun":             "drip_irrigation",
    "mini sprinkler":       "drip_irrigation",
    "irrigation system":    "drip_irrigation",
    "drip pipe":            "drip_irrigation",
    "drip line":            "drip_irrigation",
    "lateral pipe system":  "drip_irrigation",
}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience list — pass to registry.extend() or however your registry works
# ─────────────────────────────────────────────────────────────────────────────

NEW_MACHINE_PROFILES = [
    CULTIVATOR_PROFILE,
    SPRAYER_PROFILE,
    DRIP_IRRIGATION_PROFILE,
]