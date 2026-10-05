"""
utils/machine_registry.py
─────────────────────────────────────────────────────────────────────────────
Central registry for ALL farm machines supported by AgriFixAR.

This is the SINGLE SOURCE OF TRUTH for:
  • Canonical machine identifiers and aliases (used by YOLO / Flutter)
  • Human-readable names in English and Hindi
  • Machine-specific area zones (for AR overlays and step hints)
  • Machine-specific parts with area mappings
  • Machine-specific safety rules and critical components
  • Machine-specific diagnostic context injected into every Gemini prompt
  • Farmer-language landmark descriptions per machine type

Supported machines:
  tractor, harvester, thresher, submersible_pump, water_pump,
  electric_motor, power_tiller, rotavator, seed_drill, sprayer,
  chaff_cutter, sugarcane_crusher, generator, diesel_engine
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class MachineAreaZone:
    """One physical inspection zone on a machine."""
    id: str           # snake_case  e.g. "engine_compartment"
    label_en: str     # "Engine compartment"
    label_hi: str     # "इंजन का हिस्सा"
    farmer_description_en: str   # How to tell a farmer where to look
    farmer_description_hi: str


@dataclass
class MachinePart:
    """A named inspectable part on a specific machine."""
    id: str                   # snake_case e.g. "impeller"
    area_zone: str            # refers to MachineAreaZone.id
    label_en: str
    label_hi: str
    farmer_description_en: str   # plain visual description
    farmer_description_hi: str
    ar_model: str             # .obj filename for Unity AR


@dataclass
class MachineProfile:
    """Full profile of one supported farm machine."""
    machine_id: str           # canonical id
    aliases: List[str]        # alternative names / YOLO labels
    label_en: str
    label_hi: str
    category: str             # "engine_driven" | "electric" | "tractor_attachment" | "pump"

    # Inspection zones for this machine
    area_zones: List[MachineAreaZone]

    # All inspectable parts
    parts: List[MachinePart]

    # Parts whose damage immediately escalates to a mechanic
    critical_parts: List[str]   # part ids

    # Parts that are part of the fuel/ignition system (trigger no-spark rule)
    fuel_system_parts: List[str]

    # Machine-specific safety warnings always injected into prompts
    base_safety_warnings_en: List[str]
    base_safety_warnings_hi: List[str]

    # Short diagnostic context paragraph injected into every Gemini prompt
    # Explains to Gemini what this machine does and its common failure modes
    diagnostic_context: str

    # Farmer-friendly one-liner: "a large red machine used for..."
    farmer_intro_en: str
    farmer_intro_hi: str


# ─────────────────────────────────────────────────────────────────────────────
# MACHINE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

_PROFILES: List[MachineProfile] = [

    # ══════════════════════════════════════════════════════════════════════════
    # TRACTOR
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="tractor",
        aliases=["tractor", "farm tractor", "agricultural tractor", "kisan tractor",
                 "mahindra", "john deere", "sonalika", "swaraj", "eicher", "farmtrac"],
        label_en="Tractor",
        label_hi="ट्रैक्टर",
        category="engine_driven",
        farmer_intro_en="a four-wheeled farm vehicle with a large diesel engine used for ploughing, towing, and powering attachments",
        farmer_intro_hi="एक चार पहियों वाला खेती का वाहन जिसमें बड़ा डीजल इंजन होता है — जोताई, खिंचाई और उपकरण चलाने के लिए इस्तेमाल होता है",
        diagnostic_context=(
            "Tractors are diesel-engine 4-wheel farm vehicles. Common failures: battery/alternator "
            "issues (won't start), clutch wear (gear won't engage), hydraulic pump failure (lift arm won't rise), "
            "fuel filter blockage (power loss), overheating (coolant/radiator), PTO shaft problems, "
            "fan belt snap. Always check the simplest electrical/fuel items before assuming mechanical failure."
        ),
        area_zones=[
            MachineAreaZone("engine_compartment", "Engine compartment", "इंजन का हिस्सा",
                "under the metal bonnet/hood at the very front of the tractor",
                "ट्रैक्टर के बिल्कुल आगे की धातु की टोपी (बोनट) के नीचे"),
            MachineAreaZone("steering_region", "Steering / driver area", "स्टीयरिंग / चालक का क्षेत्र",
                "near the steering wheel — the round wheel the driver holds",
                "स्टीयरिंग व्हील के पास — वह गोल पहिया जिसे ड्राइवर पकड़ता है"),
            MachineAreaZone("transmission_area", "Gearbox / transmission", "गियरबॉक्स का हिस्सा",
                "the large heavy metal box directly under the driver's seat",
                "चालक की सीट के ठीक नीचे का बड़ा भारी धातु का डिब्बा"),
            MachineAreaZone("fuel_system", "Fuel system", "ईंधन प्रणाली",
                "near the fuel tank — the metal container where diesel is poured",
                "ईंधन टैंक के पास — वह धातु का कंटेनर जिसमें डीजल डाला जाता है"),
            MachineAreaZone("hydraulic_system", "Hydraulic system", "हाइड्रोलिक प्रणाली",
                "at the rear of the tractor, near the lift arm (the metal arm that raises implements)",
                "ट्रैक्टर के पिछले हिस्से में, लिफ्ट आर्म के पास — वह धातु की भुजा जो उपकरण ऊपर उठाती है"),
            MachineAreaZone("pto_area", "PTO shaft area", "PTO शाफ़्ट का हिस्सा",
                "at the very back of the tractor, the spinning metal rod that powers attachments",
                "ट्रैक्टर के बिल्कुल पीछे, वह घूमने वाली धातु की छड़ जो उपकरणों को चलाती है"),
            MachineAreaZone("undercarriage", "Undercarriage", "ट्रैक्टर का निचला हिस्सा",
                "under the belly of the tractor, between the four wheels, close to the ground",
                "ट्रैक्टर के पेट के नीचे, चारों पहियों के बीच, जमीन के करीब"),
            MachineAreaZone("wheel_area", "Wheel / tyre area", "पहिया / टायर का क्षेत्र",
                "near the large rear wheels or the smaller front wheels",
                "बड़े पिछले पहियों या छोटे अगले पहियों के पास"),
        ],
        parts=[
            MachinePart("battery_terminal", "engine_compartment", "Battery terminal", "बैटरी टर्मिनल",
                "the rectangular metal box with a thick red and thick black cable clipped to two round metal posts on top",
                "चौकोर धातु का डिब्बा जिसके ऊपर दो गोल धातु की टोटियों पर मोटी लाल और मोटी काली तार लगी हों", "battery.obj"),
            MachinePart("wiring_harness", "engine_compartment", "Wiring harness", "तार का गुच्छा",
                "the bundle of coloured wires running along the engine",
                "इंजन के साथ-साथ चलने वाले रंग-बिरंगे तारों का गुच्छा", "wiring.obj"),
            MachinePart("fan_belt", "engine_compartment", "Fan belt", "पंखे की बेल्ट",
                "the black rubber loop that runs around two or more round pulleys near the engine fan",
                "इंजन पंखे के पास दो या अधिक गोल चक्रों के आसपास लगी काली रबर की पट्टी", "fan_belt.obj"),
            MachinePart("radiator_cap", "engine_compartment", "Radiator cap", "रेडिएटर की टोपी",
                "the round cap (usually yellow or black) on top of the metal cooling tank at the front of the engine",
                "इंजन के आगे धातु के ठंडे टैंक के ऊपर गोल टोपी (आमतौर पर पीली या काली)", "radiator_cap.obj"),
            MachinePart("engine_oil_dipstick", "engine_compartment", "Engine oil dipstick", "इंजन तेल की छड़",
                "the long thin metal rod with a loop handle that you pull out to check the engine oil level",
                "लंबी पतली धातु की छड़ जिसे खींचकर इंजन का तेल जांचते हैं", "dipstick.obj"),
            MachinePart("air_filter", "engine_compartment", "Air filter", "हवा का फिल्टर",
                "the round or rectangular box (often red or grey) connected to a rubber pipe going into the engine",
                "गोल या चौकोर डिब्बा (अक्सर लाल या भूरा) जो रबर की नली से इंजन से जुड़ा होता है", "air_filter.obj"),
            MachinePart("spark_plug", "engine_compartment", "Fuel injector / glow plug", "फ्यूल इंजेक्टर",
                "the metal plug screwed into the top or side of the engine block with a thick black wire attached",
                "इंजन ब्लॉक के ऊपर या बगल में लगा धातु का प्लग जिसमें मोटी काली तार लगी हो", "spark_plug.obj"),
            MachinePart("ignition_key", "steering_region", "Ignition key", "इग्निशन चाबी",
                "the key slot near the steering wheel — the key should be in the OFF position (pointing down or left)",
                "स्टीयरिंग व्हील के पास चाबी की जगह — चाबी OFF पर होनी चाहिए", "ignition.obj"),
            MachinePart("gear_lever", "steering_region", "Gear lever", "गियर लीवर",
                "the tall metal stick near the driver's seat that is moved to change gears",
                "चालक की सीट के पास लंबी धातु की छड़ी जिसे गियर बदलने के लिए हिलाते हैं", "gear_lever.obj"),
            MachinePart("clutch_pedal", "steering_region", "Clutch pedal", "क्लच पेडल",
                "the wide metal plate on the floor near the driver's left foot",
                "चालक के बाएं पैर के पास फर्श पर चौड़ी धातु की पट्टी", "clutch_pedal.obj"),
            MachinePart("clutch_cable", "transmission_area", "Clutch cable / linkage", "क्लच केबल",
                "the metal wire or rod that connects the clutch pedal to the gearbox — runs from the pedal upward",
                "धातु की तार या छड़ जो क्लच पेडल को गियरबॉक्स से जोड़ती है", "clutch_cable.obj"),
            MachinePart("fuel_cap", "fuel_system", "Fuel tank cap", "ईंधन टैंक की टोपी",
                "the round cap on top of the fuel tank that you unscrew to pour diesel",
                "ईंधन टैंक के ऊपर गोल टोपी जिसे खोलकर डीजल डाला जाता है", "fuel_cap.obj"),
            MachinePart("fuel_filter", "fuel_system", "Fuel filter", "ईंधन फिल्टर",
                "a small transparent or metal cylinder in the fuel line between the tank and the engine",
                "ईंधन पाइप में टैंक और इंजन के बीच छोटा पारदर्शी या धातु का सिलेंडर", "fuel_filter.obj"),
            MachinePart("hydraulic_pump", "hydraulic_system", "Hydraulic pump", "हाइड्रोलिक पंप",
                "the metal pump at the rear of the tractor that powers the lifting arm",
                "ट्रैक्टर के पीछे धातु का पंप जो लिफ्ट आर्म को ऊपर उठाता है", "hydraulic_pump.obj"),
            MachinePart("drain_plug", "undercarriage", "Oil drain plug", "तेल निकासी प्लग",
                "the bolt at the very bottom of the engine or gearbox, used to drain old oil",
                "इंजन या गियरबॉक्स के बिल्कुल नीचे का बोल्ट जिससे पुराना तेल निकाला जाता है", "drain_plug.obj"),
        ],
        critical_parts=["engine_block", "crankshaft", "cylinder_head", "injection_pump"],
        fuel_system_parts=["fuel_cap", "fuel_filter", "fuel_line", "fuel_tank", "injection_pump"],
        base_safety_warnings_en=[
            "Switch off the engine and remove the key before touching any part.",
            "Never put hands near the fan belt or any spinning part — wait 2 minutes after engine off.",
            "Do not work under the tractor unless on flat ground with wheels blocked by stones.",
        ],
        base_safety_warnings_hi=[
            "कोई भी हिस्सा छूने से पहले इंजन बंद करें और चाबी निकाल लें।",
            "पंखे की बेल्ट या किसी घूमने वाले हिस्से के पास हाथ न डालें — इंजन बंद होने के 2 मिनट बाद।",
            "ट्रैक्टर के नीचे तभी काम करें जब समतल जमीन हो और पहियों के नीचे पत्थर लगे हों।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # COMBINE HARVESTER
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="harvester",
        aliases=["harvester", "combine harvester", "combine", "reaper", "wheat harvester",
                 "paddy harvester", "rice harvester", "claas", "john deere harvester",
                 "karjan", "preet harvester"],
        label_en="Combine Harvester",
        label_hi="कंबाइन हार्वेस्टर",
        category="engine_driven",
        farmer_intro_en="a large self-propelled machine that cuts, threshes, and separates grain from the crop in one pass",
        farmer_intro_hi="एक बड़ी स्वचालित मशीन जो एक ही बार में फसल काटती है, गाहती है और अनाज अलग करती है",
        diagnostic_context=(
            "Combine harvesters are complex machines with a header (cutting platform), threshing drum, "
            "separation sieves, grain tank, and straw walkers. Common failures: header blockage (crop jam), "
            "threshing concave clearance wrong (grain loss / cracking), sieve blockage (poor separation), "
            "belt/chain breaks, engine overheating from chaff clogging the radiator, "
            "hydraulic header lift failure, grain elevator blockage, and feeder house chain wear. "
            "Check header and feeder house first for blockages before checking mechanical components."
        ),
        area_zones=[
            MachineAreaZone("header_area", "Header / cutting platform", "हेडर / कटाई का हिस्सा",
                "the wide flat cutting bar at the very front of the machine, with spinning blades (reel) above it",
                "मशीन के बिल्कुल आगे चौड़ा कटाई का हिस्सा जिसके ऊपर घूमने वाला रील है"),
            MachineAreaZone("feeder_house", "Feeder house / intake", "फीडर हाउस",
                "the slanted enclosed chute between the header and the main body that carries crop inside",
                "हेडर और मुख्य शरीर के बीच तिरछी बंद नली जो फसल अंदर ले जाती है"),
            MachineAreaZone("threshing_area", "Threshing drum area", "थ्रेशिंग ड्रम का हिस्सा",
                "inside the main body, the large rotating drum that beats grain off the stalks",
                "मुख्य शरीर के अंदर बड़ा घूमने वाला ड्रम जो डंठलों से अनाज अलग करता है"),
            MachineAreaZone("sieve_area", "Cleaning / sieve area", "छलनी का हिस्सा",
                "below the threshing drum, the vibrating flat sieves that separate grain from chaff",
                "थ्रेशिंग ड्रम के नीचे कांपती हुई छलनियां जो अनाज को भूसे से अलग करती हैं"),
            MachineAreaZone("grain_tank", "Grain tank / hopper", "अनाज की टंकी",
                "the large box on top of the machine where clean grain is collected",
                "मशीन के ऊपर बड़ा डिब्बा जहां साफ अनाज इकट्ठा होता है"),
            MachineAreaZone("engine_compartment", "Engine compartment", "इंजन का हिस्सा",
                "at the rear of the machine, behind the operator cab, where the main engine is",
                "मशीन के पीछे, चालक के केबिन के पीछे, जहां मुख्य इंजन है"),
            MachineAreaZone("drive_belt_area", "Drive belts / chains", "ड्राइव बेल्ट / चेन का हिस्सा",
                "on the right side of the machine, the large rubber belts and metal chains that transfer power",
                "मशीन के दाहिनी तरफ बड़ी रबर बेल्ट और धातु की चेन जो पावर ट्रांसफर करती हैं"),
        ],
        parts=[
            MachinePart("reel", "header_area", "Reel / crop lifter", "रील",
                "the large rotating cylinder with flat bars above the cutter bar that pushes crop into the blades",
                "कटाई की पट्टी के ऊपर बड़ा घूमने वाला सिलेंडर जो फसल को ब्लेड की ओर धकेलता है", "reel.obj"),
            MachinePart("cutter_bar", "header_area", "Cutter bar / knife", "कटर बार",
                "the long horizontal row of V-shaped blades at the very bottom of the header that cut the crop",
                "हेडर के बिल्कुल नीचे V-आकार के ब्लेड की लंबी क्षैतिज पंक्ति जो फसल काटती है", "cutter_bar.obj"),
            MachinePart("feeder_chain", "feeder_house", "Feeder house chain", "फीडर चेन",
                "the metal chain with slats inside the slanted intake chute that carries crop upward",
                "तिरछी नली के अंदर स्लैट वाली धातु की चेन जो फसल को ऊपर ले जाती है", "feeder_chain.obj"),
            MachinePart("threshing_drum", "threshing_area", "Threshing drum / rotor", "थ्रेशिंग ड्रम",
                "the large rotating drum inside the machine body — it has bars or rasp bars on its surface",
                "मशीन के अंदर बड़ा घूमने वाला ड्रम — उसकी सतह पर उभरी हुई पट्टियां हैं", "threshing_drum.obj"),
            MachinePart("concave", "threshing_area", "Concave / grate", "कॉन्केव / जाली",
                "the curved metal grate below the threshing drum with holes that let grain through",
                "थ्रेशिंग ड्रम के नीचे छेद वाली मुड़ी हुई धातु की जाली जिससे अनाज गिरता है", "concave.obj"),
            MachinePart("upper_sieve", "sieve_area", "Upper sieve / chaffer", "ऊपरी छलनी",
                "the upper vibrating flat screen with adjustable louvres that separates large chaff",
                "ऊपरी कांपती हुई छलनी जिसमें समायोज्य पत्तियां होती हैं जो बड़े भूसे को अलग करती हैं", "upper_sieve.obj"),
            MachinePart("lower_sieve", "sieve_area", "Lower sieve / cleaning sieve", "निचली छलनी",
                "the lower vibrating screen with smaller holes that separates clean grain from fine chaff",
                "निचली कांपती छलनी जिसमें छोटे छेद हैं जो साफ अनाज को महीन भूसे से अलग करती है", "lower_sieve.obj"),
            MachinePart("grain_elevator", "grain_tank", "Grain elevator / auger", "अनाज एलिवेटर",
                "the metal tube with a rotating screw inside that lifts clean grain up into the grain tank",
                "धातु की नली जिसके अंदर घूमने वाला पेंच है जो साफ अनाज को ऊपर टंकी में ले जाता है", "grain_elevator.obj"),
            MachinePart("radiator_cap", "engine_compartment", "Radiator cap", "रेडिएटर की टोपी",
                "the round cap on top of the cooling tank — check if chaff is blocking the radiator mesh",
                "कूलिंग टैंक के ऊपर गोल टोपी — जांचें कि भूसे से रेडिएटर जाली बंद तो नहीं है", "radiator_cap.obj"),
            MachinePart("main_drive_belt", "drive_belt_area", "Main drive belt", "मुख्य ड्राइव बेल्ट",
                "the widest rubber belt on the right side of the machine that transfers engine power to the threshing drum",
                "मशीन के दाहिनी तरफ सबसे चौड़ी रबर बेल्ट जो इंजन की शक्ति थ्रेशिंग ड्रम को देती है", "main_belt.obj"),
        ],
        critical_parts=["threshing_drum", "concave", "feeder_chain"],
        fuel_system_parts=["fuel_cap", "fuel_filter", "fuel_line", "fuel_tank"],
        base_safety_warnings_en=[
            "NEVER open inspection covers while the machine is running — rotating drums and chains cause severe injuries.",
            "Disengage all drives and shut off the engine before clearing any blockage.",
            "Let the machine run empty for 30 seconds before shutting off to clear remaining crop.",
        ],
        base_safety_warnings_hi=[
            "मशीन चलते समय कभी भी ढक्कन न खोलें — घूमने वाले ड्रम और चेन से गंभीर चोट लग सकती है।",
            "किसी भी जाम को साफ करने से पहले सभी ड्राइव बंद करें और इंजन बंद करें।",
            "बंद करने से पहले मशीन को 30 सेकंड खाली चलाएं ताकि बची फसल साफ हो जाए।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # THRESHER (stationary)
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="thresher",
        aliases=["thresher", "threshing machine", "grain thresher", "wheat thresher",
                 "paddy thresher", "axial flow thresher", "multicrop thresher", "thrashing machine"],
        label_en="Thresher / Threshing Machine",
        label_hi="थ्रेशर / गाहने की मशीन",
        category="engine_driven",
        farmer_intro_en="a stationary machine that beats harvested crop bundles to separate grain from stalks and chaff",
        farmer_intro_hi="एक स्थिर मशीन जो कटी हुई फसल की पूलियों को पीटकर डंठलों और भूसे से अनाज अलग करती है",
        diagnostic_context=(
            "Stationary threshers are powered by a tractor PTO or a separate diesel/electric motor. "
            "Common failures: threshing cylinder blockage (crop overload), concave/drum clearance wrong "
            "(grain cracking or high loss), belt/flat belt slip or break, bearing failure (loud knocking), "
            "sieve blockage (grain in bhusa), vibration due to unbalanced drum, "
            "feed roller jam, and drive pulley misalignment. "
            "PTO-driven: check PTO shaft connection first. Motor-driven: check belt tension and motor current."
        ),
        area_zones=[
            MachineAreaZone("feed_inlet", "Feed inlet / throat", "फीड इनलेट",
                "the wide opening at the front/top where crop bundles are fed in by hand",
                "आगे/ऊपर का चौड़ा मुंह जहां हाथ से फसल की पूलियां डाली जाती हैं"),
            MachineAreaZone("threshing_chamber", "Threshing cylinder/drum", "थ्रेशिंग सिलेंडर",
                "inside the machine, the main rotating drum with metal bars that beats the crop",
                "मशीन के अंदर मुख्य घूमने वाला ड्रम जिसमें धातु की पट्टियां हैं जो फसल पीटती हैं"),
            MachineAreaZone("sieve_outlet", "Sieve / cleaning outlet", "छलनी / सफाई निकास",
                "the lower front area where grain falls through sieves and comes out",
                "निचला आगे का हिस्सा जहां से अनाज छलनी से गुजरकर निकलता है"),
            MachineAreaZone("bhusa_outlet", "Straw / bhusa outlet", "भूसा निकास",
                "the rear opening or blower fan area where straw and chaff blow out",
                "पिछला मुंह या ब्लोअर पंखा जहां से डंठल और भूसा उड़कर निकलता है"),
            MachineAreaZone("drive_side", "Drive / belt side", "ड्राइव / बेल्ट की साइड",
                "the right or left side panel where the flat belt, pulleys, and bearings are visible",
                "दाहिनी या बाईं साइड का पैनल जहां फ्लैट बेल्ट, पुली और बेयरिंग दिखती हैं"),
            MachineAreaZone("engine_compartment", "Engine / motor area", "इंजन / मोटर का हिस्सा",
                "the separate diesel engine or electric motor that powers the thresher (if not PTO-driven)",
                "अलग डीजल इंजन या बिजली की मोटर जो थ्रेशर को चलाती है (अगर PTO से नहीं चलता)"),
        ],
        parts=[
            MachinePart("feed_roller", "feed_inlet", "Feed roller", "फीड रोलर",
                "the pair of rubber-covered rollers just inside the feed opening that grip and pull crop in",
                "फीड मुंह के ठीक अंदर रबर से ढके रोलर जो फसल को खींचते हैं", "feed_roller.obj"),
            MachinePart("threshing_cylinder", "threshing_chamber", "Threshing cylinder / drum", "थ्रेशिंग सिलेंडर",
                "the main large rotating drum with metal teeth/bars — the heart of the machine",
                "मुख्य बड़ा घूमने वाला ड्रम जिसमें धातु के दांत/पट्टियां हैं — मशीन का दिल", "threshing_drum.obj"),
            MachinePart("concave_plate", "threshing_chamber", "Concave / grate plate", "कॉन्केव प्लेट",
                "the curved metal grate that wraps around the drum — grain falls through its holes",
                "ड्रम के चारों ओर मुड़ी हुई धातु की जाली — अनाज उसके छेदों से गिरता है", "concave.obj"),
            MachinePart("main_flat_belt", "drive_side", "Main flat belt / V-belt", "मुख्य फ्लैट बेल्ट",
                "the wide flat rubber belt or V-belt that runs from the engine pulley to the drum pulley",
                "चौड़ी फ्लैट रबर बेल्ट या V-बेल्ट जो इंजन पुली से ड्रम पुली तक जाती है", "flat_belt.obj"),
            MachinePart("drum_bearing", "drive_side", "Drum shaft bearing", "ड्रम बेयरिंग",
                "the round metal bearing block at each end of the drum shaft — makes a grinding noise when worn",
                "ड्रम शाफ़्ट के दोनों सिरों पर गोल धातु का बेयरिंग ब्लॉक — घिसने पर पीसने की आवाज़ आती है", "bearing.obj"),
            MachinePart("cleaning_sieve", "sieve_outlet", "Cleaning sieve / jali", "सफाई छलनी",
                "the flat or slightly curved metal screen with small holes that grain passes through — check for blockage",
                "सपाट या थोड़ी मुड़ी धातु की जाली जिसमें छोटे छेद हैं जिनसे अनाज गुजरता है — जाम की जांच करें", "cleaning_sieve.obj"),
            MachinePart("blower_fan", "bhusa_outlet", "Blower fan / air fan", "ब्लोअर पंखा",
                "the large rotating fan inside the machine that blows chaff and straw out of the bhusa outlet",
                "मशीन के अंदर बड़ा घूमने वाला पंखा जो भूसे को बाहर उड़ाता है", "blower_fan.obj"),
        ],
        critical_parts=["threshing_cylinder", "drum_bearing", "concave_plate"],
        fuel_system_parts=["fuel_cap", "fuel_filter", "fuel_line", "fuel_tank"],
        base_safety_warnings_en=[
            "NEVER put hands inside the feed opening while the machine is running — the drum will cause amputation.",
            "Stand to the side of the feed opening, never directly in front.",
            "Stop the machine completely and wait for the drum to stop before clearing any jam.",
        ],
        base_safety_warnings_hi=[
            "मशीन चलते समय फीड मुंह में हाथ कभी न डालें — ड्रम हाथ काट सकता है।",
            "फीड मुंह के बगल में खड़े हों, सीधे सामने कभी नहीं।",
            "कोई जाम साफ करने से पहले मशीन पूरी तरह बंद करें और ड्रम को रुकने दें।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SUBMERSIBLE PUMP
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="submersible_pump",
        aliases=["submersible pump", "submersible", "borewell pump", "borewell motor",
                 "submersible motor", "underground pump", "tubewell pump", "patal pump",
                 "borewell", "tubewell"],
        label_en="Submersible Pump / Borewell Motor",
        label_hi="सबमर्सिबल पंप / बोरवेल मोटर",
        category="electric",
        farmer_intro_en="an electric pump installed deep inside a borewell or well to lift groundwater to the surface",
        farmer_intro_hi="एक बिजली का पंप जो बोरवेल या कुएं के अंदर गहराई में लगाया जाता है और जमीन का पानी ऊपर खींचता है",
        diagnostic_context=(
            "Submersible pumps run on 3-phase or single-phase electricity. The motor and impeller are "
            "submerged in water inside the borewell. Common failures: no water output (dry run, impeller wear, "
            "low water table), low pressure/flow (worn impeller, partially blocked inlet strainer, "
            "undersized pipe), motor burnout (single-phasing, dry run, overloading), "
            "starter/DOL/star-delta panel tripping, capacitor failure (single-phase), "
            "delivery pipe blockage or leak, and check valve failure (water falls back). "
            "NEVER run without water — motor burns in seconds. All diagnostics are done at surface level "
            "(control panel, starter panel, riser pipe) — the motor itself is underground."
        ),
        area_zones=[
            MachineAreaZone("control_panel", "Starter / control panel", "स्टार्टर / कंट्रोल पैनल",
                "the metal box mounted on the wall or pole with switches, fuses, and indicator lights — where you start the pump",
                "दीवार या खंभे पर लगा धातु का बक्सा जिसमें स्विच, फ्यूज़ और संकेत बल्ब हैं — यहाँ से पंप चालू होता है"),
            MachineAreaZone("delivery_pipe", "Delivery / riser pipe", "डिलीवरी / राइज़र पाइप",
                "the pipe that comes out of the borewell above ground and carries water to the field or tank",
                "बोरवेल से जमीन के ऊपर निकलने वाला पाइप जो पानी खेत या टंकी तक ले जाता है"),
            MachineAreaZone("wellhead", "Wellhead / borewell top", "बोरवेल का मुंह",
                "the top opening of the borewell where the delivery pipe and electric cable come out",
                "बोरवेल का ऊपरी मुंह जहां से डिलीवरी पाइप और बिजली की तार निकलती है"),
            MachineAreaZone("electrical_connection", "Electrical wiring / connection", "बिजली की तारें",
                "the wires connecting the control panel to the electricity supply and down into the borewell",
                "कंट्रोल पैनल को बिजली आपूर्ति से और बोरवेल के अंदर तक जोड़ने वाली तारें"),
        ],
        parts=[
            MachinePart("starter_panel", "control_panel", "DOL / star-delta starter panel", "स्टार्टर पैनल",
                "the metal box with the main ON/OFF switch, overload relay (small adjustable dial), and three-phase fuses",
                "मुख्य ON/OFF स्विच, ओवरलोड रिले (छोटी समायोज्य डायल) और तीन-चरण फ्यूज़ वाला धातु का बक्सा", "starter_panel.obj"),
            MachinePart("capacitor", "control_panel", "Capacitor (single-phase)", "कैपेसिटर",
                "a small cylindrical metal or plastic can inside the control panel — used in single-phase pumps only",
                "कंट्रोल पैनल के अंदर छोटा बेलनाकार धातु या प्लास्टिक का डिब्बा — केवल सिंगल-फेज पंप में", "capacitor.obj"),
            MachinePart("overload_relay", "control_panel", "Overload relay / thermal relay", "ओवरलोड रिले",
                "a small device inside the starter panel with a red reset button — trips to protect the motor from overheating",
                "स्टार्टर पैनल के अंदर लाल रीसेट बटन वाला छोटा उपकरण — मोटर को गर्म होने से बचाने के लिए ट्रिप करता है", "overload_relay.obj"),
            MachinePart("main_fuses", "control_panel", "Main fuses / MCB", "मुख्य फ्यूज़",
                "the three cylindrical glass or ceramic fuses (or MCB switches) inside the panel that protect against short circuit",
                "पैनल के अंदर तीन बेलनाकार कांच या सिरेमिक फ्यूज़ (या MCB स्विच) जो शॉर्ट सर्किट से बचाते हैं", "fuse.obj"),
            MachinePart("delivery_pipe_joint", "delivery_pipe", "Delivery pipe joint / elbow", "डिलीवरी पाइप जोड़",
                "the threaded joints or rubber couplings connecting sections of the above-ground delivery pipe",
                "जमीन के ऊपर डिलीवरी पाइप के हिस्सों को जोड़ने वाले थ्रेडेड जोड़ या रबर कपलिंग", "pipe_joint.obj"),
            MachinePart("check_valve", "delivery_pipe", "Check valve / non-return valve", "चेक वाल्व",
                "a metal or plastic valve in the delivery pipe that lets water flow only one way (up) — prevents water falling back",
                "डिलीवरी पाइप में धातु या प्लास्टिक का वाल्व जो पानी को केवल एक दिशा (ऊपर) जाने देता है", "check_valve.obj"),
            MachinePart("pump_cable", "electrical_connection", "Submersible cable", "सबमर्सिबल केबल",
                "the flat 3-core waterproof cable that runs from the control panel down into the borewell to the motor",
                "कंट्रोल पैनल से बोरवेल के अंदर मोटर तक जाने वाली सपाट 3-कोर वाटरप्रूफ तार", "pump_cable.obj"),
        ],
        critical_parts=["submersible_motor", "pump_shaft"],
        fuel_system_parts=[],   # electric — no fuel system
        base_safety_warnings_en=[
            "SWITCH OFF the main power supply before touching any wiring — electric shock from wet hands is fatal.",
            "NEVER run the pump without water (dry run) — the motor burns out in seconds.",
            "Test voltage at the panel before assuming a motor fault — low voltage causes most motor burnouts.",
        ],
        base_safety_warnings_hi=[
            "कोई भी तार छूने से पहले मुख्य बिजली आपूर्ति बंद करें — गीले हाथों से बिजली का झटका जानलेवा है।",
            "पंप को कभी भी पानी के बिना (ड्राई रन) न चलाएं — मोटर तुरंत जल जाती है।",
            "मोटर की खराबी मानने से पहले पैनल पर वोल्टेज जांचें — कम वोल्टेज से अधिकांश मोटर जलती हैं।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SURFACE / CENTRIFUGAL WATER PUMP
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="water_pump",
        aliases=["water pump", "centrifugal pump", "surface pump", "monoblock pump",
                 "diesel pump", "pump set", "irrigation pump", "pani pump", "monoblock"],
        label_en="Water Pump / Pump Set",
        label_hi="पानी का पंप / पंप सेट",
        category="engine_driven",
        farmer_intro_en="a surface-mounted pump (diesel or electric) used to lift water from a canal, pond, or open well for irrigation",
        farmer_intro_hi="एक जमीन पर लगा पंप (डीजल या बिजली) जो नहर, तालाब या कुएं से सिंचाई के लिए पानी ऊपर उठाता है",
        diagnostic_context=(
            "Surface centrifugal pump sets are either diesel-engine driven or electric-motor driven. "
            "Common failures: pump not priming (air leak in suction pipe, foot valve stuck/clogged), "
            "low discharge (worn impeller, partially open valve, suction pipe too long/narrow), "
            "no discharge (impeller jam, wrong rotation direction for electric), "
            "cavitation noise (suction head too high, air leaks), mechanical seal leak (water at shaft), "
            "engine issues (fuel, air filter, choke). "
            "Always check suction pipe, foot valve, and priming before assuming pump/engine fault."
        ),
        area_zones=[
            MachineAreaZone("pump_body", "Pump casing / volute", "पंप का शरीर",
                "the main metal body of the pump — usually painted, with the suction pipe on one side and discharge on top",
                "पंप का मुख्य धातु का शरीर — आमतौर पर रंगा हुआ, एक तरफ सक्शन पाइप और ऊपर डिस्चार्ज"),
            MachineAreaZone("suction_side", "Suction pipe / inlet", "सक्शन पाइप / इनलेट",
                "the pipe that goes from the pump down into the water source (canal/pond/well) — black rubber or PVC",
                "वह पाइप जो पंप से नीचे पानी के स्रोत (नहर/तालाब/कुएं) में जाता है — काली रबर या PVC"),
            MachineAreaZone("discharge_side", "Discharge pipe / outlet", "डिस्चार्ज पाइप / आउटलेट",
                "the pipe carrying pressurized water away from the pump to the field — usually goes upward",
                "पंप से दबाव में पानी खेत की ओर ले जाने वाला पाइप — आमतौर पर ऊपर की ओर जाता है"),
            MachineAreaZone("engine_compartment", "Engine / motor", "इंजन / मोटर",
                "the diesel engine or electric motor bolted to the same base plate as the pump",
                "पंप के साथ एक ही बेस प्लेट पर बोल्ट किया डीजल इंजन या बिजली की मोटर"),
            MachineAreaZone("coupling_area", "Coupling / shaft area", "कपलिंग / शाफ़्ट का हिस्सा",
                "the connection between the engine/motor shaft and the pump shaft — usually a flexible rubber coupling",
                "इंजन/मोटर शाफ़्ट और पंप शाफ़्ट के बीच का कनेक्शन — आमतौर पर लचीली रबर कपलिंग"),
        ],
        parts=[
            MachinePart("foot_valve", "suction_side", "Foot valve / strainer", "फुट वाल्व",
                "the metal cage with a flap valve at the very bottom of the suction pipe, submerged in the water source",
                "सक्शन पाइप के बिल्कुल नीचे पानी के स्रोत में डूबा धातु का पिंजरा जिसमें फ्लैप वाल्व है", "foot_valve.obj"),
            MachinePart("suction_pipe_joint", "suction_side", "Suction pipe joints", "सक्शन पाइप जोड़",
                "all threaded or rubber-clamped joints along the suction pipe — any tiny air leak stops the pump from priming",
                "सक्शन पाइप के सभी थ्रेडेड या रबर-क्लैम्प वाले जोड़ — छोटी हवा की लीक भी पंप को प्राइम होने से रोकती है", "pipe_joint.obj"),
            MachinePart("priming_plug", "pump_body", "Priming plug / priming hole", "प्राइमिंग प्लग",
                "the small round plug or cap on the top of the pump body that you remove to pour water in for priming",
                "पंप शरीर के ऊपर छोटा गोल प्लग या टोपी जिसे हटाकर प्राइमिंग के लिए पानी डाला जाता है", "priming_plug.obj"),
            MachinePart("mechanical_seal", "pump_body", "Mechanical seal / gland", "मैकेनिकल सील",
                "where the pump shaft exits the casing — a small gap here leaks water if the seal is worn",
                "जहां पंप का शाफ़्ट आवरण से बाहर निकलता है — सील घिसने पर यहाँ से पानी टपकता है", "mechanical_seal.obj"),
            MachinePart("impeller", "pump_body", "Impeller", "इम्पेलर",
                "the internal spinning wheel with curved blades inside the pump that moves the water — not visible without opening",
                "पंप के अंदर घुमावदार पत्तियों वाला घूमने वाला पहिया जो पानी चलाता है — खोले बिना दिखता नहीं", "impeller.obj"),
            MachinePart("discharge_valve", "discharge_side", "Gate valve / discharge valve", "गेट वाल्व",
                "the hand-operated wheel valve on the discharge pipe that you turn to control water flow",
                "डिस्चार्ज पाइप पर हाथ से चलाने वाला पहिया वाल्व जिसे पानी का बहाव नियंत्रित करने के लिए घुमाते हैं", "gate_valve.obj"),
            MachinePart("fuel_cap", "engine_compartment", "Fuel cap / tank", "ईंधन टैंक",
                "the fuel tank cap on the diesel engine — check fuel level before diagnosing power issues",
                "डीजल इंजन पर ईंधन टैंक की टोपी — पावर की समस्या जांचने से पहले ईंधन स्तर जांचें", "fuel_cap.obj"),
            MachinePart("air_filter", "engine_compartment", "Air filter", "हवा का फिल्टर",
                "the round or rectangular box connected to the engine intake — check if clogged with dust",
                "इंजन के इनटेक से जुड़ा गोल या चौकोर डिब्बा — जांचें कि धूल से बंद तो नहीं है", "air_filter.obj"),
        ],
        critical_parts=["impeller", "mechanical_seal"],
        fuel_system_parts=["fuel_cap", "fuel_filter", "fuel_line", "fuel_tank"],
        base_safety_warnings_en=[
            "NEVER prime the pump while it is running — fill water through the priming plug only when stopped.",
            "Keep hands away from the coupling and rotating shaft at all times.",
            "Do not run the pump dry even for a few seconds — the mechanical seal will overheat and crack.",
        ],
        base_safety_warnings_hi=[
            "पंप चलते समय प्राइमिंग कभी न करें — बंद होने पर ही प्राइमिंग प्लग से पानी डालें।",
            "कपलिंग और घूमने वाले शाफ़्ट से हाथ हमेशा दूर रखें।",
            "पंप को कुछ सेकंड के लिए भी सूखा न चलाएं — मैकेनिकल सील गर्म होकर टूट जाएगी।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ELECTRIC MOTOR (standalone)
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="electric_motor",
        aliases=["electric motor", "motor", "induction motor", "3 phase motor",
                 "single phase motor", "motor pump", "bijli motor", "electric engine"],
        label_en="Electric Motor",
        label_hi="बिजली की मोटर",
        category="electric",
        farmer_intro_en="an electric motor used to drive pumps, threshers, chaff cutters, or other farm machinery",
        farmer_intro_hi="एक बिजली की मोटर जो पंप, थ्रेशर, चारा काटने की मशीन या अन्य कृषि उपकरण चलाती है",
        diagnostic_context=(
            "Electric motors fail due to: single phasing (one phase missing — motor hums but won't start), "
            "overloading (thermal protector trips, motor hot), capacitor failure (single-phase won't start), "
            "winding burnout (burning smell, no rotation), bearing failure (loud grinding/vibration), "
            "starting capacitor vs run capacitor confusion, wrong voltage supply, "
            "belt/load mechanical jam preventing start. "
            "Diagnose by checking supply voltage first, then starter/overload, then capacitor, then windings."
        ),
        area_zones=[
            MachineAreaZone("motor_body", "Motor body / frame", "मोटर का शरीर",
                "the cylindrical metal body of the motor with cooling fins on the outside",
                "मोटर का बेलनाकार धातु का शरीर जिसके बाहर ठंडा करने के लिए पसलियां (फिन्स) होती हैं"),
            MachineAreaZone("terminal_box", "Terminal box / connection box", "टर्मिनल बॉक्स",
                "the small metal box on the side of the motor where the power supply wires connect",
                "मोटर के बगल में छोटा धातु का बक्सा जहां बिजली की आपूर्ति की तारें जुड़ती हैं"),
            MachineAreaZone("starter_panel", "Starter / control panel", "स्टार्टर / कंट्रोल पैनल",
                "the panel or box on the wall with switches, fuses, overload relay, and capacitor",
                "दीवार पर बक्सा जिसमें स्विच, फ्यूज़, ओवरलोड रिले और कैपेसिटर होते हैं"),
            MachineAreaZone("drive_end", "Drive end / pulley end", "ड्राइव एंड / पुली एंड",
                "the end of the motor where the shaft sticks out and the belt pulley or coupling is attached",
                "मोटर का वह सिरा जहां से शाफ़्ट बाहर निकलता है और बेल्ट पुली या कपलिंग लगी होती है"),
        ],
        parts=[
            MachinePart("capacitor", "starter_panel", "Run/start capacitor", "कैपेसिटर",
                "a cylindrical metal can (single-phase motors only) — if bulging or leaking, it has failed",
                "बेलनाकार धातु का डिब्बा (केवल सिंगल-फेज मोटर में) — फूला हुआ या लीक हो तो खराब है", "capacitor.obj"),
            MachinePart("overload_relay", "starter_panel", "Overload relay / thermal relay", "ओवरलोड रिले",
                "small device with a red reset button inside the panel — if tripped, the motor won't start until reset",
                "पैनल के अंदर लाल रीसेट बटन वाला छोटा उपकरण — ट्रिप होने पर रीसेट तक मोटर नहीं चलेगी", "overload_relay.obj"),
            MachinePart("main_fuses", "starter_panel", "Supply fuses / MCB", "आपूर्ति फ्यूज़",
                "the fuses or MCB switches that protect the motor supply — check all three phases are live",
                "मोटर आपूर्ति की सुरक्षा करने वाले फ्यूज़ या MCB स्विच — जांचें कि तीनों फेज़ चालू हैं", "fuse.obj"),
            MachinePart("motor_bearings", "drive_end", "Motor bearings", "मोटर बेयरिंग",
                "the internal bearings at both ends of the motor shaft — grinding or squealing noise means worn bearings",
                "मोटर शाफ़्ट के दोनों सिरों पर अंदरूनी बेयरिंग — पीसने या चीखने की आवाज़ का मतलब घिसी बेयरिंग", "bearing.obj"),
            MachinePart("terminal_connections", "terminal_box", "Winding terminal connections", "टर्मिनल कनेक्शन",
                "the six (or three) screw terminals inside the connection box where supply wires attach",
                "कनेक्शन बॉक्स के अंदर छह (या तीन) स्क्रू टर्मिनल जहां आपूर्ति तारें जुड़ती हैं", "terminal.obj"),
            MachinePart("drive_pulley", "drive_end", "Drive pulley / shaft coupling", "ड्राइव पुली",
                "the round grooved wheel on the motor shaft end that the belt sits in — check for belt alignment",
                "मोटर शाफ़्ट के सिरे पर गोल खांचेदार पहिया जिसमें बेल्ट बैठती है — बेल्ट की सीध जांचें", "pulley.obj"),
        ],
        critical_parts=["motor_windings", "motor_shaft"],
        fuel_system_parts=[],
        base_safety_warnings_en=[
            "SWITCH OFF and LOCK OUT the main power before touching any part of the motor or wiring.",
            "Test voltage with a meter — never touch bare wires to test if power is present.",
            "A humming motor that won't turn could be single-phasing — check all three fuses before investigating further.",
        ],
        base_safety_warnings_hi=[
            "मोटर या तारें छूने से पहले मुख्य बिजली बंद करें और लॉक करें।",
            "वोल्टेज मीटर से जांचें — बिजली है या नहीं देखने के लिए नंगी तारें कभी न छुएं।",
            "जो मोटर गुनगुनाती है पर नहीं घूमती वह सिंगल-फेजिंग हो सकती है — आगे जांच से पहले तीनों फ्यूज़ जांचें।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # POWER TILLER
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="power_tiller",
        aliases=["power tiller", "walking tractor", "hand tractor", "mini tractor",
                 "power tiller machine", "VST shakti", "kubota power tiller", "chhota tractor"],
        label_en="Power Tiller",
        label_hi="पावर टिलर",
        category="engine_driven",
        farmer_intro_en="a two-wheeled walk-behind machine with a diesel engine used for tilling, puddling, and carrying loads",
        farmer_intro_hi="एक दो पहियों वाली पैदल चलाने की मशीन जिसमें डीजल इंजन होता है — जोताई, कीचड़ बनाने और ढुलाई के लिए",
        diagnostic_context=(
            "Power tillers use a single-cylinder diesel engine directly connected to a gearbox. "
            "Common failures: hard starting (decompression lever, glow plug, fuel issue), "
            "engine stalls under load (fuel filter, governor spring), "
            "tine/rotavator attachment jam (stones in soil), "
            "steering handle vibration (tine bearing, gear shaft), "
            "clutch not disengaging (cable stretch, friction plate wear), "
            "engine overheating (air-cooled — check cooling fins for mud blockage), "
            "oil leak from gear housing."
        ),
        area_zones=[
            MachineAreaZone("engine_compartment", "Engine / motor area", "इंजन का हिस्सा",
                "the single-cylinder diesel engine at the top-front of the power tiller",
                "पावर टिलर के ऊपर-आगे का एकल सिलेंडर डीजल इंजन"),
            MachineAreaZone("gearbox_area", "Gearbox / transmission", "गियरबॉक्स का हिस्सा",
                "the large metal housing below the engine that contains the gears and connects to the wheels or tines",
                "इंजन के नीचे बड़ा धातु का आवरण जिसमें गियर हैं और जो पहियों या टाइन से जुड़ता है"),
            MachineAreaZone("handle_area", "Steering handles / controls", "स्टीयरिंग हैंडल / नियंत्रण",
                "the two long metal handles at the rear that the farmer holds to guide the machine",
                "पीछे दो लंबे धातु के हैंडल जिन्हें किसान मशीन चलाने के लिए पकड़ता है"),
            MachineAreaZone("attachment_area", "Tine / rotary attachment", "टाइन / रोटरी अटैचमेंट",
                "the rotating metal blades/tines at the very bottom that dig into the soil",
                "बिल्कुल नीचे घूमने वाली धातु की पत्तियां/टाइन जो मिट्टी में खुदाई करती हैं"),
            MachineAreaZone("fuel_system", "Fuel system", "ईंधन प्रणाली",
                "the small fuel tank on top of the engine and the fuel pipes leading to the injector",
                "इंजन के ऊपर छोटा ईंधन टैंक और इंजेक्टर तक जाने वाली ईंधन पाइपें"),
        ],
        parts=[
            MachinePart("decompression_lever", "engine_compartment", "Decompression lever", "डीकंप्रेशन लीवर",
                "a small lever on the engine that you flip up before pulling the starter rope — releases compression to make starting easier",
                "इंजन पर छोटा लीवर जिसे स्टार्टर रस्सी खींचने से पहले ऊपर उठाते हैं — कंप्रेशन छोड़ता है", "decomp_lever.obj"),
            MachinePart("starter_rope", "engine_compartment", "Starter rope / recoil starter", "स्टार्टर रस्सी",
                "the rope on the side of the engine that you pull firmly to crank and start the engine",
                "इंजन के बगल में रस्सी जिसे इंजन चालू करने के लिए जोर से खींचते हैं", "starter_rope.obj"),
            MachinePart("air_filter", "engine_compartment", "Air filter", "हवा का फिल्टर",
                "the round metal or plastic box with an oil-bath or dry element connected to the air intake",
                "हवा के इनटेक से जुड़ा गोल धातु या प्लास्टिक का बक्सा — तेल-स्नान या सूखा एलिमेंट", "air_filter.obj"),
            MachinePart("fuel_cap", "fuel_system", "Fuel tank cap", "ईंधन टैंक की टोपी",
                "the cap on the small fuel tank — check fuel level and condition (diesel, not petrol)",
                "छोटे ईंधन टैंक की टोपी — ईंधन स्तर और गुणवत्ता जांचें (डीजल, पेट्रोल नहीं)", "fuel_cap.obj"),
            MachinePart("clutch_lever", "handle_area", "Clutch lever / engagement lever", "क्लच लीवर",
                "the hand lever on one of the handles — squeeze to engage drive, release to stop the tines/wheels",
                "एक हैंडल पर हाथ का लीवर — ड्राइव लगाने के लिए दबाएं, टाइन/पहिए रोकने के लिए छोड़ें", "clutch_lever.obj"),
            MachinePart("gear_selector", "gearbox_area", "Gear selector / speed lever", "गियर सेलेक्टर",
                "the lever near the gearbox that selects forward/reverse and speed — check it is fully in one position",
                "गियरबॉक्स के पास लीवर जो आगे/पीछे और गति चुनता है — जांचें कि पूरी तरह एक स्थिति में है", "gear_selector.obj"),
            MachinePart("tine_bolts", "attachment_area", "Tine attachment bolts", "टाइन के बोल्ट",
                "the large bolts that hold the rotating tines/blades to the shaft — check for looseness or breakage",
                "घूमने वाले टाइन/पत्तियों को शाफ़्ट से जोड़ने वाले बड़े बोल्ट — ढीलेपन या टूटने की जांच करें", "tine_bolt.obj"),
        ],
        critical_parts=["crankshaft", "connecting_rod", "gearbox_shaft"],
        fuel_system_parts=["fuel_cap", "fuel_filter", "fuel_line", "fuel_tank"],
        base_safety_warnings_en=[
            "NEVER reach toward the rotating tines — they will cause severe cuts and cannot be stopped instantly.",
            "Disengage the clutch lever before changing gear or adjusting attachments.",
            "On slopes, never leave the power tiller unattended — engage the parking lock.",
        ],
        base_safety_warnings_hi=[
            "घूमते टाइन की ओर कभी हाथ न बढ़ाएं — वे गंभीर कट कर सकते हैं और तुरंत नहीं रुकते।",
            "गियर बदलने या अटैचमेंट समायोजित करने से पहले क्लच लीवर छोड़ें।",
            "ढलान पर पावर टिलर को अकेला कभी न छोड़ें — पार्किंग लॉक लगाएं।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # CHAFF CUTTER (Toka / Tokan machine)
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="chaff_cutter",
        aliases=["chaff cutter", "toka machine", "tokan machine", "fodder cutter",
                 "hay cutter", "bhusa machine", "chara cutter", "green fodder cutter"],
        label_en="Chaff Cutter / Fodder Cutter",
        label_hi="चारा काटने की मशीन / टोका मशीन",
        category="electric",
        farmer_intro_en="a machine that chops straw, green fodder, or dry crop stalks into small pieces for cattle feed",
        farmer_intro_hi="एक मशीन जो पुआल, हरा चारा या सूखे डंठलों को पशुओं के चारे के लिए छोटे टुकड़ों में काटती है",
        diagnostic_context=(
            "Chaff cutters are either electric-motor driven or tractor-PTO driven. "
            "Common failures: blades become dull (fibrous cut, not clean chop), "
            "flywheel/drum jam (overfeeding or hard stalks), "
            "belt slip (reduced speed, incomplete cuts), "
            "blade bolt loose/missing (vibration, noise), "
            "feed roller jam (crop compaction), "
            "electric motor tripping (single-phasing, overload from dull blades or overfeeding), "
            "safety guard missing or bypass (serious injury risk). "
            "HIGHEST INJURY RISK machine — many hand amputations recorded in India."
        ),
        area_zones=[
            MachineAreaZone("feed_inlet", "Feed inlet / throat", "फीड इनलेट",
                "the rectangular opening at the front where fodder is fed in — EXTREME DANGER ZONE",
                "आगे का चौकोर मुंह जहां चारा डाला जाता है — अत्यंत खतरनाक क्षेत्र"),
            MachineAreaZone("cutting_drum", "Cutting drum / flywheel", "कटिंग ड्रम / फ्लाईव्हील",
                "inside the machine, the heavy rotating drum or flywheel with blades attached",
                "मशीन के अंदर भारी घूमने वाला ड्रम या फ्लाईव्हील जिसमें ब्लेड लगे हैं"),
            MachineAreaZone("discharge_chute", "Discharge chute / outlet", "डिस्चार्ज चूट",
                "the chute where chopped fodder comes out — usually at the side or top",
                "कटा हुआ चारा निकलने वाली चूट — आमतौर पर बगल में या ऊपर"),
            MachineAreaZone("drive_side", "Motor / belt / drive side", "मोटर / बेल्ट / ड्राइव साइड",
                "the side of the machine where the motor or PTO shaft, belt, and pulleys are located",
                "मशीन का वह हिस्सा जहां मोटर या PTO शाफ़्ट, बेल्ट और पुली होती हैं"),
        ],
        parts=[
            MachinePart("cutter_blades", "cutting_drum", "Cutter blades", "काटने के ब्लेड",
                "the sharp metal blades bolted to the rotating drum — check for dullness, cracks, or loose bolts",
                "घूमने वाले ड्रम पर बोल्ट की तेज धातु की पत्तियां — सुस्तीपन, दरारें या ढीले बोल्ट जांचें", "cutter_blades.obj"),
            MachinePart("feed_roller", "feed_inlet", "Feed rollers", "फीड रोलर",
                "the pair of rubber-topped rollers just inside the feed opening that grip and pull fodder in at a set speed",
                "फीड मुंह के अंदर रबर वाले रोलर जो चारे को एक निश्चित गति से खींचते हैं", "feed_roller.obj"),
            MachinePart("drive_belt", "drive_side", "Drive belt / V-belt", "ड्राइव बेल्ट",
                "the V-belt or flat belt connecting the motor/PTO to the cutting drum pulley",
                "मोटर/PTO को कटिंग ड्रम पुली से जोड़ने वाली V-बेल्ट या फ्लैट बेल्ट", "v_belt.obj"),
            MachinePart("safety_guard", "feed_inlet", "Feed inlet safety guard", "सुरक्षा गार्ड",
                "the metal or plastic guard over the feed opening that prevents hands reaching the blades — MUST be in place",
                "फीड मुंह के ऊपर धातु या प्लास्टिक का गार्ड जो हाथों को ब्लेड तक पहुंचने से रोकता है — जरूर लगा होना चाहिए", "safety_guard.obj"),
        ],
        critical_parts=["cutter_blades", "flywheel", "cutting_drum"],
        fuel_system_parts=[],
        base_safety_warnings_en=[
            "CRITICAL: NEVER put hands inside the feed opening — blades cause instant amputation.",
            "The safety guard over the feed inlet MUST be in place before any operation.",
            "Stop the machine and wait for the drum to completely stop before clearing any jam.",
            "Never operate a chaff cutter alone — always have a second person present.",
        ],
        base_safety_warnings_hi=[
            "अत्यंत महत्वपूर्ण: फीड मुंह में हाथ कभी न डालें — ब्लेड तुरंत हाथ काट देते हैं।",
            "किसी भी काम से पहले फीड इनलेट के ऊपर सुरक्षा गार्ड जरूर लगा होना चाहिए।",
            "कोई जाम साफ करने से पहले मशीन बंद करें और ड्रम के पूरी तरह रुकने का इंतजार करें।",
            "चारा काटने की मशीन कभी अकेले न चलाएं — हमेशा दूसरा व्यक्ति पास होना चाहिए।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # DIESEL ENGINE (standalone stationary)
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="diesel_engine",
        aliases=["diesel engine", "stationary engine", "engine", "kerosene engine",
                 "lister engine", "kirloskar engine", "lombardini", "yanmar",
                 "standalone engine", "pumpset engine"],
        label_en="Diesel Engine (Stationary)",
        label_hi="डीजल इंजन (स्थिर)",
        category="engine_driven",
        farmer_intro_en="a stationary diesel engine used to drive pumps, threshers, or other farm equipment via a belt or shaft",
        farmer_intro_hi="एक स्थिर डीजल इंजन जो बेल्ट या शाफ़्ट के माध्यम से पंप, थ्रेशर या अन्य कृषि उपकरण चलाता है",
        diagnostic_context=(
            "Stationary diesel engines (1-3 cylinder, 3-20 HP) power pumps, threshers, and other equipment. "
            "Common failures: hard starting (decompression, fuel, glow plug, air filter), "
            "engine stalls under load (governor spring, fuel filter, injection timing), "
            "excessive smoke — black (overload/injector), blue (oil burning), white (water/head gasket), "
            "overheating (air-cooled: mud on fins; water-cooled: coolant level), "
            "knocking (low oil, worn bearing, injection timing), "
            "engine won't reach full RPM (governor, fuel delivery). "
        ),
        area_zones=[
            MachineAreaZone("engine_compartment", "Main engine body", "इंजन का मुख्य शरीर",
                "the main cast-iron engine block with cylinder head on top",
                "ऊपर सिलेंडर हेड के साथ मुख्य ढली-लोहे का इंजन ब्लॉक"),
            MachineAreaZone("fuel_system", "Fuel system", "ईंधन प्रणाली",
                "the fuel tank, fuel filter, injection pump, and injector on the engine",
                "इंजन पर ईंधन टैंक, ईंधन फिल्टर, इंजेक्शन पंप और इंजेक्टर"),
            MachineAreaZone("cooling_area", "Cooling fins / radiator", "ठंडा करने का हिस्सा",
                "the large metal fins on the cylinder barrel (air-cooled) or the water radiator (water-cooled)",
                "सिलेंडर बैरल पर बड़ी धातु की पसलियां (एयर-कूल्ड) या पानी का रेडिएटर (वॉटर-कूल्ड)"),
            MachineAreaZone("exhaust_area", "Exhaust / smoke pipe", "एग्जॉस्ट / धुएं की पाइप",
                "the metal pipe that carries exhaust gases away from the engine — observe smoke colour",
                "धातु की पाइप जो इंजन से निकास गैसें बाहर ले जाती है — धुएं का रंग देखें"),
            MachineAreaZone("drive_side", "Drive pulley / PTO area", "ड्राइव पुली / PTO का हिस्सा",
                "the flywheel side of the engine where the drive pulley or belt wheel is mounted",
                "इंजन का फ्लाईव्हील वाला हिस्सा जहां ड्राइव पुली या बेल्ट व्हील लगा है"),
            MachineAreaZone("lubrication", "Oil / lubrication system", "तेल / स्नेहन प्रणाली",
                "the oil filler cap, dipstick, and drain plug for the engine lubricating oil",
                "इंजन के लुब्रिकेटिंग तेल के लिए तेल भरने की टोपी, डिपस्टिक और ड्रेन प्लग"),
        ],
        parts=[
            MachinePart("fuel_filter", "fuel_system", "Fuel filter / sediment bowl", "ईंधन फिल्टर",
                "a small glass bowl or metal cylinder on the fuel line between the tank and the injection pump",
                "टैंक और इंजेक्शन पंप के बीच ईंधन पाइप पर छोटा कांच का कटोरा या धातु का सिलेंडर", "fuel_filter.obj"),
            MachinePart("air_filter", "engine_compartment", "Air filter / air cleaner", "हवा का फिल्टर",
                "the metal or plastic box (often with an oil bath) connected to the air intake — check for mud/dust blockage",
                "हवा के इनटेक से जुड़ा धातु या प्लास्टिक का बक्सा (अक्सर तेल-स्नान के साथ) — कीचड़/धूल जाम जांचें", "air_filter.obj"),
            MachinePart("engine_oil_dipstick", "lubrication", "Engine oil dipstick", "इंजन तेल की छड़",
                "the long thin metal rod you pull out to check the engine oil level — minimum level mark must be covered",
                "लंबी पतली धातु की छड़ जिसे इंजन तेल का स्तर जांचने के लिए खींचते हैं — न्यूनतम स्तर की निशान ढकी होनी चाहिए", "dipstick.obj"),
            MachinePart("cooling_fins", "cooling_area", "Cooling fins (air-cooled)", "ठंडक की पसलियां",
                "the many thin metal fins on the cylinder barrel — check if clogged with mud, chaff, or dry grass",
                "सिलेंडर बैरल पर पतली धातु की पसलियां — जांचें कि कीचड़, भूसे या सूखी घास से बंद तो नहीं हैं", "cooling_fins.obj"),
            MachinePart("exhaust_pipe", "exhaust_area", "Exhaust pipe / silencer", "एग्जॉस्ट पाइप",
                "the metal pipe and silencer — observe smoke colour: black=overload, blue=oil burning, white=coolant leak",
                "धातु की पाइप और साइलेंसर — धुएं का रंग देखें: काला=अधिभार, नीला=तेल जल रहा, सफेद=कूलेंट लीक", "exhaust.obj"),
            MachinePart("decompression_lever", "engine_compartment", "Decompression lever", "डीकंप्रेशन लीवर",
                "small lever on the cylinder head — flip before pulling the start handle to release compression",
                "सिलेंडर हेड पर छोटा लीवर — स्टार्ट हैंडल खींचने से पहले कंप्रेशन छोड़ने के लिए पलटें", "decomp_lever.obj"),
        ],
        critical_parts=["crankshaft", "cylinder_head", "injection_pump"],
        fuel_system_parts=["fuel_cap", "fuel_filter", "fuel_line", "fuel_tank", "injection_pump"],
        base_safety_warnings_en=[
            "Never use the decompression lever while the engine is running — this will cause a kickback injury.",
            "Let the engine cool for 10 minutes before touching the cylinder head or exhaust pipe — severe burn risk.",
            "Keep hands away from the drive pulley and belt at all times — never attempt to stop by hand.",
        ],
        base_safety_warnings_hi=[
            "इंजन चलते समय डीकंप्रेशन लीवर कभी न छुएं — किकबैक से चोट लग सकती है।",
            "सिलेंडर हेड या एग्जॉस्ट पाइप छूने से पहले इंजन को 10 मिनट ठंडा होने दें — जलने का खतरा।",
            "ड्राइव पुली और बेल्ट से हाथ हमेशा दूर रखें — हाथ से रोकने की कोशिश कभी न करें।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # ROTAVATOR (tractor attachment)
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="rotavator",
        aliases=["rotavator", "rotary tiller", "rotary cultivator", "rotary tiller attachment",
                 "rota", "rotovator", "soil tiller"],
        label_en="Rotavator / Rotary Tiller",
        label_hi="रोटावेटर / रोटरी टिलर",
        category="tractor_attachment",
        farmer_intro_en="a tractor-mounted attachment that uses rotating blades to till and break up soil, connected to the tractor PTO",
        farmer_intro_hi="एक ट्रैक्टर से जुड़ा उपकरण जो घूमने वाले ब्लेड से मिट्टी जोतता और तोड़ता है — ट्रैक्टर PTO से जुड़ा होता है",
        diagnostic_context=(
            "Rotavators are PTO-driven tractor attachments. Main gearbox receives power from tractor PTO shaft. "
            "Common failures: blade breakage (stones), blade bolt shearing (overload), "
            "gearbox oil leak (seal failure), gearbox noise (bearing wear, gear tooth damage), "
            "PTO shaft coupling damage (misalignment), rotor shaft bent (large stone impact), "
            "blade holder (flange) cracked, shear bolt in PTO shaft has sheared (overload protection). "
            "Check PTO shaft and shear bolt first — shear bolt is designed to break to protect the gearbox."
        ),
        area_zones=[
            MachineAreaZone("pto_shaft", "PTO shaft connection", "PTO शाफ़्ट कनेक्शन",
                "the rotating shaft that connects from the tractor PTO to the rotavator gearbox",
                "घूमने वाला शाफ़्ट जो ट्रैक्टर PTO से रोटावेटर गियरबॉक्स तक जोड़ता है"),
            MachineAreaZone("gearbox_area", "Rotavator gearbox", "रोटावेटर गियरबॉक्स",
                "the metal gearbox at the top-centre of the rotavator that transfers power to the rotor",
                "रोटावेटर के ऊपर-बीच में धातु का गियरबॉक्स जो रोटर को शक्ति देता है"),
            MachineAreaZone("rotor_area", "Rotor / blade shaft", "रोटर / ब्लेड शाफ़्ट",
                "the horizontal shaft running the full width of the rotavator that carries all the blades",
                "रोटावेटर की पूरी चौड़ाई में चलने वाला क्षैतिज शाफ़्ट जिसमें सभी ब्लेड लगे हैं"),
            MachineAreaZone("blade_area", "Tilling blades / L-blades", "जोताई के ब्लेड",
                "the curved L-shaped or J-shaped metal blades bolted to the rotor that dig into the soil",
                "रोटर पर बोल्ट किए मुड़े हुए L-आकार या J-आकार के धातु के ब्लेड जो मिट्टी में खोदते हैं"),
            MachineAreaZone("side_gearbox", "Side gearbox / chain box", "साइड गियरबॉक्स / चेन बॉक्स",
                "the gearbox at one or both ends of the rotor shaft — contains bevel gears or chains",
                "रोटर शाफ़्ट के एक या दोनों सिरों पर गियरबॉक्स — जिसमें बेवल गियर या चेन होती हैं"),
        ],
        parts=[
            MachinePart("shear_bolt", "pto_shaft", "PTO shear bolt / safety bolt", "शियर बोल्ट",
                "the deliberately weak bolt in the PTO shaft that breaks when the rotavator hits a large stone — protects the gearbox",
                "PTO शाफ़्ट में जानबूझकर कमज़ोर बोल्ट जो बड़े पत्थर से टकराने पर टूटता है — गियरबॉक्स को बचाता है", "shear_bolt.obj"),
            MachinePart("pto_coupling", "pto_shaft", "PTO coupling / universal joint", "PTO कपलिंग",
                "the cross-shaped joint that allows the PTO shaft to flex and turn at angles",
                "क्रॉस के आकार का जोड़ जो PTO शाफ़्ट को लचकने और कोणों पर घूमने देता है", "pto_coupling.obj"),
            MachinePart("gearbox_oil_level", "gearbox_area", "Gearbox oil level", "गियरबॉक्स तेल स्तर",
                "the oil level sight glass or dipstick on the main gearbox — low oil causes gear and bearing damage",
                "मुख्य गियरबॉक्स पर तेल स्तर की साइट ग्लास या डिपस्टिक — कम तेल से गियर और बेयरिंग खराब होती है", "gearbox_oil.obj"),
            MachinePart("rotor_blades", "blade_area", "Rotor blades", "रोटर ब्लेड",
                "the L-shaped metal blades — check for cracks, bending, or missing blades on each flange",
                "L-आकार के धातु के ब्लेड — हर फ्लैंज पर दरारें, मुड़ना, या गायब ब्लेड जांचें", "rotor_blade.obj"),
            MachinePart("blade_bolt", "blade_area", "Blade attachment bolt", "ब्लेड का बोल्ट",
                "the bolt that holds each blade to the blade holder/flange — check tightness and presence of all bolts",
                "हर ब्लेड को ब्लेड होल्डर/फ्लैंज से जोड़ने वाला बोल्ट — सभी बोल्ट की कसाई और उपस्थिति जांचें", "blade_bolt.obj"),
        ],
        critical_parts=["rotor_shaft", "main_gearbox_casing"],
        fuel_system_parts=[],
        base_safety_warnings_en=[
            "DISENGAGE PTO and raise the rotavator before reversing the tractor — never reverse with PTO engaged.",
            "NEVER stand behind or beside the rotavator during operation — thrown stones cause serious injury.",
            "Wait for all blades to stop completely before inspecting — they can coast for 30+ seconds after PTO off.",
        ],
        base_safety_warnings_hi=[
            "पलटाई से पहले PTO हटाएं और रोटावेटर ऊपर उठाएं — PTO लगे होने पर पीछे कभी न जाएं।",
            "चालू होने के दौरान रोटावेटर के पीछे या बगल में कभी न खड़े हों — उड़ते पत्थर गंभीर चोट करते हैं।",
            "जांच करने से पहले सभी ब्लेड को पूरी तरह रुकने दें — PTO बंद होने के 30+ सेकंड बाद तक घूम सकते हैं।",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # GENERATOR (farm / portable)
    # ══════════════════════════════════════════════════════════════════════════
    MachineProfile(
        machine_id="generator",
        aliases=["generator", "genset", "diesel generator", "petrol generator",
                 "portable generator", "inverter generator", "electric generator",
                 "bijli generator", "light plant"],
        label_en="Generator / Genset",
        label_hi="जनरेटर / जेनसेट",
        category="engine_driven",
        farmer_intro_en="a petrol or diesel engine connected to an alternator that produces electricity for the farm or home",
        farmer_intro_hi="एक पेट्रोल या डीजल इंजन जो अल्टरनेटर से जुड़कर खेत या घर के लिए बिजली बनाता है",
        diagnostic_context=(
            "Farm generators are 1-10 kVA, single-phase, petrol or diesel. "
            "Common failures: no output voltage (AVR failure, capacitor failure, winding fault, speed too low), "
            "engine starts but no power (excitation capacitor failed, AVR failed), "
            "engine won't start (fuel, air filter, choke position, spark plug for petrol), "
            "overload tripping (breaker, too many appliances), "
            "unstable frequency/voltage (governor, carburetor, speed), "
            "battery not charging on electric start (diode/rectifier, alternator). "
            "Check engine first, then electrical output."
        ),
        area_zones=[
            MachineAreaZone("engine_compartment", "Engine", "इंजन",
                "the petrol or diesel engine — the noisy part that runs when the generator is on",
                "पेट्रोल या डीजल इंजन — वह शोरगुल वाला हिस्सा जो जनरेटर चालू होने पर चलता है"),
            MachineAreaZone("alternator_end", "Alternator / generator end", "अल्टरनेटर / जनरेटर एंड",
                "the cylindrical metal housing at the end of the machine opposite the engine — this produces electricity",
                "इंजन के विपरीत सिरे पर बेलनाकार धातु का आवरण — यह बिजली पैदा करता है"),
            MachineAreaZone("control_panel", "Control panel / output panel", "कंट्रोल पैनल / आउटपुट पैनल",
                "the panel with output sockets, main circuit breaker, voltmeter, and frequency meter",
                "आउटपुट सॉकेट, मुख्य सर्किट ब्रेकर, वोल्टमीटर और फ्रीक्वेंसी मीटर वाला पैनल"),
            MachineAreaZone("fuel_system", "Fuel tank / fuel system", "ईंधन टैंक / ईंधन प्रणाली",
                "the fuel tank and fuel tap/valve — check fuel level and that the fuel tap is open",
                "ईंधन टैंक और ईंधन नल/वाल्व — ईंधन स्तर और नल खुला होना जांचें"),
        ],
        parts=[
            MachinePart("avr", "alternator_end", "AVR — Automatic Voltage Regulator", "AVR",
                "a small circuit board (usually green) inside the alternator cover that controls the output voltage",
                "अल्टरनेटर कवर के अंदर छोटा सर्किट बोर्ड (आमतौर पर हरा) जो आउटपुट वोल्टेज नियंत्रित करता है", "avr.obj"),
            MachinePart("excitation_capacitor", "alternator_end", "Excitation capacitor", "एक्साइटेशन कैपेसिटर",
                "a round or oval metal can inside the alternator end — if bulging, it has failed and there will be no output",
                "अल्टरनेटर एंड के अंदर गोल या अंडाकार धातु का डिब्बा — फूला हो तो खराब है और आउटपुट नहीं आएगा", "capacitor.obj"),
            MachinePart("main_breaker", "control_panel", "Main circuit breaker / MCB", "मुख्य सर्किट ब्रेकर",
                "the large switch on the output panel — if tripped, it pops up and must be reset before power comes",
                "आउटपुट पैनल पर बड़ा स्विच — ट्रिप होने पर उठ जाता है और पावर आने से पहले रीसेट करना होगा", "breaker.obj"),
            MachinePart("fuel_tap", "fuel_system", "Fuel tap / petcock", "ईंधन नल",
                "the small lever or valve under the fuel tank that must be in the OPEN position for fuel to flow",
                "ईंधन टैंक के नीचे छोटा लीवर या वाल्व जो OPEN स्थिति में होना चाहिए ताकि ईंधन बह सके", "fuel_tap.obj"),
            MachinePart("air_filter", "engine_compartment", "Air filter", "हवा का फिल्टर",
                "the box or foam element on the engine air intake — clean or replace if clogged with dust",
                "इंजन हवा के इनटेक पर बक्सा या फोम एलिमेंट — धूल से बंद हो तो साफ करें या बदलें", "air_filter.obj"),
            MachinePart("spark_plug", "engine_compartment", "Spark plug (petrol) / glow plug (diesel)", "स्पार्क प्लग",
                "the plug screwed into the top of the engine cylinder — black and sooty means rich mixture or oil fouling",
                "इंजन सिलेंडर के ऊपर लगा प्लग — काला और कालिख से ढका मतलब गाढ़ा मिश्रण या तेल", "spark_plug.obj"),
        ],
        critical_parts=["alternator_windings", "avr"],
        fuel_system_parts=["fuel_cap", "fuel_tap", "fuel_filter", "fuel_line", "fuel_tank"],
        base_safety_warnings_en=[
            "NEVER run a generator indoors or in a partially enclosed area — carbon monoxide is odourless and kills silently.",
            "Let the generator cool for 2 minutes before refuelling — spilling fuel on a hot engine causes fire.",
            "Do not overload the generator — adding too many appliances will damage the AVR and alternator.",
        ],
        base_safety_warnings_hi=[
            "जनरेटर कभी भी घर के अंदर या आधे बंद जगह पर न चलाएं — कार्बन मोनोऑक्साइड बेरंग-बेगंध है और चुपचाप मारती है।",
            "ईंधन भरने से पहले जनरेटर को 2 मिनट ठंडा होने दें — गर्म इंजन पर ईंधन गिरने से आग लग सकती है।",
            "जनरेटर पर अधिक भार न डालें — बहुत सारे उपकरण लगाने से AVR और अल्टरनेटर खराब होंगे।",
        ],
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Merge new machines from machine_registry_additions.py
# ─────────────────────────────────────────────────────────────────────────────
# The additions file stores profiles as plain dicts for portability.
# Here we convert each dict → MachineProfile + MachineAreaZone + MachinePart
# and append them to _PROFILES so every public API function sees them.

def _profile_from_dict(d: dict) -> MachineProfile:
    """Convert a machine_registry_additions profile dict to a MachineProfile."""
    return MachineProfile(
        machine_id            = d["machine_id"],
        aliases               = list(d.get("aliases", [d["machine_id"]])),
        label_en              = d["label_en"],
        label_hi              = d["label_hi"],
        category              = d.get("category", "other"),
        farmer_intro_en       = d.get("farmer_intro_en", ""),
        farmer_intro_hi       = d.get("farmer_intro_hi", ""),
        diagnostic_context    = d.get("diagnostic_context", ""),
        critical_parts        = d.get("critical_parts", []),
        fuel_system_parts     = d.get("fuel_system_parts", []),
        base_safety_warnings_en = d.get("base_safety_warnings_en", []),
        base_safety_warnings_hi = d.get("base_safety_warnings_hi", []),
        area_zones = [
            MachineAreaZone(
                id                    = z["id"],
                label_en              = z["label_en"],
                label_hi              = z["label_hi"],
                farmer_description_en = z.get("farmer_description_en", ""),
                farmer_description_hi = z.get("farmer_description_hi", ""),
            )
            for z in d.get("area_zones", [])
        ],
        parts = [
            MachinePart(
                id                    = p["id"],
                area_zone             = p.get("area_zone", ""),
                label_en              = p.get("label_en", ""),
                label_hi              = p.get("label_hi", ""),
                farmer_description_en = p.get("farmer_description_en", ""),
                farmer_description_hi = p.get("farmer_description_hi", ""),
                ar_model              = p.get("ar_model", "part.obj"),
            )
            for p in d.get("parts", [])
        ],
    )


try:
    from .machine_registry_additions import NEW_MACHINE_PROFILES, NEW_MACHINE_ALIASES
    for _raw in NEW_MACHINE_PROFILES:
        # Skip if this machine_id is already registered (idempotent on reload)
        if _raw["machine_id"] not in {p.machine_id for p in _PROFILES}:
            _PROFILES.append(_profile_from_dict(_raw))
except ImportError:
    # Additions file missing — registry still fully functional for existing machines
    NEW_MACHINE_ALIASES = {}
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "⚠️  machine_registry_additions.py not found — "
        "cultivator/sprayer/drip_irrigation profiles not loaded."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lookup indexes (built once at import time, AFTER all profiles are merged)
# ─────────────────────────────────────────────────────────────────────────────

_ID_INDEX: Dict[str, MachineProfile] = {p.machine_id: p for p in _PROFILES}
_ALIAS_INDEX: Dict[str, MachineProfile] = {}

for _profile in _PROFILES:
    for _alias in _profile.aliases:
        _ALIAS_INDEX[_alias.lower()] = _profile

# Also register the string-keyed aliases from the additions file
for _alias_str, _canonical_id in NEW_MACHINE_ALIASES.items():
    if _canonical_id in _ID_INDEX:
        _ALIAS_INDEX[_alias_str.lower()] = _ID_INDEX[_canonical_id]

# Build the global part→area map across ALL machines (including new ones)
_GLOBAL_PART_AREA_MAP: Dict[str, str] = {}
for _profile in _PROFILES:
    for _part in _profile.parts:
        _GLOBAL_PART_AREA_MAP[_part.id] = _part.area_zone


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_profile(machine_type: str) -> Optional[MachineProfile]:
    """
    Return the MachineProfile for a given machine identifier or alias.
    Returns None if the machine is not recognised.
    Case-insensitive.
    """
    key = machine_type.strip().lower()
    return _ID_INDEX.get(key) or _ALIAS_INDEX.get(key)


def get_profile_or_default(machine_type: str) -> MachineProfile:
    """
    Like get_profile() but returns the tractor profile as a safe default
    if the machine type is not recognised.
    """
    return get_profile(machine_type) or _ID_INDEX["tractor"]

def get_shutdown_instruction(machine_type: str) -> dict:
    """
    Return the correct shutdown instruction for any machine type.
    Derives this from the machine profile's category and properties.
    Never hardcodes machine names — uses category-based logic.
    """
    profile = get_profile_or_default(machine_type)
    category = profile.category.lower() if profile else ""
    machine_id = machine_type.lower()
    
    # Category-based rules — works for ANY machine in that category
    if "electric" in category or "motor" in category or machine_id in _ELECTRIC_MACHINES:
        return {
            "action": "power_off",
            "instruction_en": (
                "Switch off the MCB / main breaker and verify all indicator "
                "lights are OFF. Wait 60 seconds for capacitors to discharge "
                "before touching any terminals."
            ),
            "instruction_hi": (
                "MCB / main switch band karein aur ensure karein ki saare "
                "indicator lights band ho jaayein. Capacitor discharge ke "
                "liye 60 second rukein."
            ),
            "required_part": "main_switch",
            "area_hint": "control_panel",
        }
    
    if "diesel" in category or "engine" in category or machine_id in _DIESEL_MACHINES:
        return {
            "action": "engine_off",
            "instruction_en": (
                "Turn off the engine and remove the ignition key. Engage "
                "the parking brake and wait for all moving parts to stop completely."
            ),
            "instruction_hi": (
                "Engine band karein aur ignition key nikaal lein. Parking "
                "brake lagayein aur saare parts band hone tak wait karein."
            ),
            "required_part": "ignition_key",
            "area_hint": "operator_cab",
        }
    
    if "hydraulic" in category or "pump" in machine_id:
        return {
            "action": "depressurise",
            "instruction_en": (
                "Lower all hydraulic implements to the ground and turn off "
                "the engine. Open the hydraulic pressure relief valve to "
                "depressurise the system."
            ),
            "instruction_hi": (
                "Saare hydraulic implements zameen par lower karein aur engine "
                "band karein. Hydraulic pressure relief valve khol ke system "
                "ko depressurise karein."
            ),
            "required_part": "hydraulic_relief_valve",
            "area_hint": "hydraulic_panel",
        }
    
    # Default: generic shutdown for any machine
    return {
        "action": "engine_off",
        "instruction_en": (
            "Ensure the machine is completely shut down and isolated from "
            "its power source before proceeding with any inspection or repair."
        ),
        "instruction_hi": (
            "Koi bhi kaam shuru karne se pehle machine ko power source se "
            "bilkul alag karein."
        ),
        "required_part": "ignition_key",
        "area_hint": "operator_cab",
    }


# Category membership — derived from machine registry, not hardcoded
_ELECTRIC_MACHINES: set = set()
_DIESEL_MACHINES: set = set()

def _init_machine_categories():
    """Populate category sets from the machine registry at startup."""
    global _ELECTRIC_MACHINES, _DIESEL_MACHINES
    from utils.machine_registry import list_supported_machines
    for m in list_supported_machines():
        mid = m["machine_id"]
        cat = m.get("category", "").lower()
        if "electric" in cat or "motor" in cat:
            _ELECTRIC_MACHINES.add(mid)
        if "diesel" in cat or "engine" in cat or "tractor" in cat or "harvester" in cat:
            _DIESEL_MACHINES.add(mid)


def resolve_machine_id(machine_type: str) -> str:
    """
    Normalise any machine name/alias to its canonical machine_id.
    Returns original string if not recognised (so it still passes to Gemini).
    """
    profile = get_profile(machine_type)
    return profile.machine_id if profile else machine_type.lower()


def list_supported_machines() -> List[Dict]:
    """Return a summary list of all supported machines (for /machines endpoint)."""
    return [
        {
            "machine_id": p.machine_id,
            "label_en": p.label_en,
            "label_hi": p.label_hi,
            "category": p.category,
            "aliases": p.aliases,
        }
        for p in _PROFILES
    ]


def get_part_area(part_id: str, machine_type: Optional[str] = None) -> str:
    """
    Return the area_zone for a given part_id.
    If machine_type is provided, checks that machine's parts first (more accurate).
    Falls back to the global map, then to 'engine_compartment'.
    """
    if machine_type:
        profile = get_profile(machine_type)
        if profile:
            for part in profile.parts:
                if part.id == part_id:
                    return part.area_zone
    return _GLOBAL_PART_AREA_MAP.get(part_id, "engine_compartment")


def get_all_part_ids(machine_type: Optional[str] = None) -> List[str]:
    """Return all known part IDs, optionally filtered to a specific machine."""
    if machine_type:
        profile = get_profile(machine_type)
        if profile:
            return [p.id for p in profile.parts]
    return list(_GLOBAL_PART_AREA_MAP.keys())


def get_area_zones(machine_type: str) -> List[MachineAreaZone]:
    """Return all area zones for a machine type."""
    profile = get_profile(machine_type)
    return profile.area_zones if profile else []


def get_safety_warnings(machine_type: str, language: str = "en") -> List[str]:
    """Return base safety warnings for a machine in the requested language."""
    profile = get_profile(machine_type)
    if not profile:
        return []
    return profile.base_safety_warnings_en if language != "hi" else profile.base_safety_warnings_hi


def get_diagnostic_context(machine_type: str) -> str:
    """Return the machine-specific diagnostic context for injection into Gemini prompts."""
    profile = get_profile(machine_type)
    return profile.diagnostic_context if profile else ""


def get_farmer_intro(machine_type: str, language: str = "en") -> str:
    """Return a farmer-friendly one-liner describing the machine."""
    profile = get_profile(machine_type)
    if not profile:
        return machine_type
    return profile.farmer_intro_en if language != "hi" else profile.farmer_intro_hi


def get_critical_parts(machine_type: str) -> List[str]:
    """Return list of critical part IDs for a machine."""
    profile = get_profile(machine_type)
    return profile.critical_parts if profile else []


def get_fuel_system_parts(machine_type: str) -> List[str]:
    """Return fuel system part IDs for a machine (for safety rule injection)."""
    profile = get_profile(machine_type)
    return profile.fuel_system_parts if profile else []


def is_electric_machine(machine_type: str) -> bool:
    """True if the machine is electrically driven (affects safety rules)."""
    profile = get_profile(machine_type)
    return profile.category == "electric" if profile else False


def is_tractor_attachment(machine_type: str) -> bool:
    """True if the machine is a tractor-mounted attachment."""
    profile = get_profile(machine_type)
    return profile.category == "tractor_attachment" if profile else False


def get_area_label(machine_type: str, area_id: str, language: str = "en") -> str:
    """Return the human-readable area label for a zone ID."""
    profile = get_profile(machine_type)
    if profile:
        for zone in profile.area_zones:
            if zone.id == area_id:
                return zone.label_en if language != "hi" else zone.label_hi
    return area_id.replace("_", " ").title()


def get_area_farmer_description(machine_type: str, area_id: str, language: str = "en") -> str:
    """Return the farmer-friendly description of where a zone is."""
    profile = get_profile(machine_type)
    if profile:
        for zone in profile.area_zones:
            if zone.id == area_id:
                return zone.farmer_description_en if language != "hi" else zone.farmer_description_hi
    return area_id.replace("_", " ")


def get_allowed_area_ids(machine_type: str) -> List[str]:
    """Return all valid area_hint values for a specific machine."""
    profile = get_profile(machine_type)
    if profile:
        return [z.id for z in profile.area_zones]
    # fallback to generic set
    return ["engine_compartment", "fuel_system", "steering_region",
            "transmission_area", "undercarriage", "wheel_area", "dashboard"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-optimised helpers  (used by services to reduce token count)
# ─────────────────────────────────────────────────────────────────────────────

def get_compact_diagnostic_hint(machine_type: str) -> str:
    """
    Return a single-line triage order for this machine type.
    Used in agent prompt instead of the verbose 8-item logic block.
    ~10-20 tokens vs ~192 tokens.
    """
    _hints = {
        "tractor":          "Triage: fuel→air_filter→battery→clutch→hydraulic→mechanical.",
        "harvester":        "Triage: crop_jam→belt/chain→concave_clearance→sieve→bearings.",
        "thresher":         "Triage: crop_jam→belt→concave→sieve→bearings→PTO_shaft.",
        "submersible_pump": "Triage: power_supply→fuses→overload_relay→voltage→cable→motor.",
        "water_pump":       "Triage: priming→foot_valve→suction_leak→impeller→seal→engine.",
        "electric_motor":   "Triage: power_supply→fuses→overload_relay→capacitor→windings→bearings.",
        "power_tiller":     "Triage: fuel→decompression→air_filter→clutch_lever→gear→tines.",
        "chaff_cutter":     "Triage: safety_guard→belt→blade_sharpness→feed_roller→motor.",
        "diesel_engine":    "Triage: fuel→air_filter→decompression→oil→cooling_fins→injection.",
        "rotavator":        "Triage: shear_bolt→PTO_shaft→gearbox_oil→blades→blade_bolts.",
        "generator":        "Triage: fuel_tap→engine→main_breaker→capacitor→AVR→windings.",
    }
    profile = get_profile(machine_type)
    mid = profile.machine_id if profile else machine_type.lower()
    return _hints.get(mid, f"Triage: external_visual→fuel/power→mechanical→internal.")


def get_compact_safety_keywords(machine_type: str) -> str:
    """
    One-line safety keywords for this machine — injected into agent prompt
    instead of the full safety_warnings sentences (~59 tok → ~15 tok).
    Already-enforced by safety_rules.py; this is just a reminder.
    """
    _keywords = {
        "tractor":          "engine_off+key_out; no_hands_near_belt/fan; chock_wheels.",
        "harvester":        "PTO_off+covers_closed; no_clearing_while_running; run_empty_30s.",
        "thresher":         "drum_stopped_before_clearing; stand_beside_not_in_front.",
        "submersible_pump": "main_power_OFF; no_dry_run; test_voltage_first.",
        "water_pump":       "no_dry_run; hands_clear_of_coupling; prime_only_when_stopped.",
        "electric_motor":   "main_power_LOCKOUT; use_meter_not_bare_hands.",
        "power_tiller":     "disengage_clutch_before_adjust; never_grab_tines.",
        "chaff_cutter":     "GUARD_IN_PLACE; never_hands_in_feed; two_person_rule.",
        "diesel_engine":    "cool_10min_before_touching; no_decomp_while_running.",
        "rotavator":        "PTO_off+raise_before_reverse; blades_coast_30s.",
        "generator":        "NEVER_indoors; cool_2min_before_refuel; no_overload.",
    }
    profile = get_profile(machine_type)
    mid = profile.machine_id if profile else machine_type.lower()
    return _keywords.get(mid, "engine/power_off_before_touching_any_part.")


def get_compact_parts_list(machine_type: str, max_parts: int = 10) -> str:
    """
    Comma-separated list of part IDs, capped to avoid bloat.
    Uses the most diagnostic-relevant parts first (order as defined in registry).
    """
    parts = get_all_part_ids(machine_type)[:max_parts]
    remaining = len(get_all_part_ids(machine_type)) - max_parts
    suffix = f"+{remaining}_more" if remaining > 0 else ""
    return ", ".join(parts) + suffix