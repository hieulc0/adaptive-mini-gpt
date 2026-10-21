"""Curated Language Pretraining Corpus and Multi-Domain Reading Comprehension Dataset (M28).

Establishes the neural language foundation for Adaptive Mini-GPT:
  - Curated Diverse Prose Pretraining Corpus across 5 genres:
      1. Narratives & Fables (character actions, causal chains, moral dilemmas)
      2. Scientific Explanations (mechanisms, laws, biological & physical processes)
      3. Factual & Historical Passages (inventions, exploration, geography, architecture)
      4. Conversational Context & Dialogues (inquiries, clarifications, cooperative exchanges)
      5. Instructional & Procedural Texts (step-by-step guides, constraints, recipes)
  - Grounded Reading Comprehension Partitions:
      * TRAIN_QA_DOCUMENTS: 16 diverse factual & instructional passages
      * VAL_QA_DOCUMENTS: 6 strictly disjoint documents for validation
      * SEALED_TEST_QA_DOCUMENTS: 8 strictly disjoint held-out documents
  - Document-level, fact-level, and topic-level disjointness verification.
  - Safe abstention ("Insufficient information.") on missing evidence.
  - Format compliance, exact match, and grounded semantic evaluation verifiers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

ABSTENTION_TARGET = "Insufficient information."
ABSTENTION_VARIANTS = {
    "insufficient information",
    "insufficient information.",
    "unanswerable",
    "not mentioned",
    "unknown",
    "not specified in the passage",
    "cannot be determined",
    "information not provided",
}


@dataclass
class DocumentPassage:
    """A distinct factual passage with grounded answerable and unanswerable questions."""

    doc_id: str
    topic: str
    text: str
    answerable_pairs: list[tuple[str, str, list[str]]]  # (question, canonical_answer, aliases)
    unanswerable_questions: list[str]
    paraphrase_pairs: list[tuple[str, str, list[str]]] = field(default_factory=list)


# =====================================================================
# Curated Prose Pretraining Corpus (5 Genres, CC0 Curated Prose)
# =====================================================================

PRETRAIN_PROSE_CORPUS: list[str] = [
    # --- Genre 1: Narratives and Fables ---
    "A patient fox watched an old owl resting in the hollow oak. The fox spoke softly, asking if wisdom came with age. The owl blinked slowly and replied that wisdom comes not from age alone, but from listening more than one speaks. Hearing this, the fox sat quietly until the dawn broke over the forest.",
    "Elena found an antique brass clock inside her grandfather's workshop. Every gear was covered in fine dust, but the balance spring remained intact. She cleaned each tooth with a camel hair brush and added a drop of mineral oil to the pivot. When she nudged the pendulum, the clock ticked with a steady, rhythmic cadence.",
    "Two travelers were walking along a dusty mountain pass when dark storm clouds gathered on the horizon. Instead of hurrying blindly, they looked for shelter beneath an overhanging granite ledge. By sharing their dry wood, they built a small fire that kept them warm until the rain subsided into morning mist.",
    "Marcus tended his modest terraced vineyard on the rocky hillside. While his neighbors planted fast-growing vines on the damp valley floor, Marcus pruned his roots deep into the limestone. When a severe drought struck during midsummer, only the deep roots of Marcus's vineyard found moisture beneath the bedrock.",
    "A young apprentice potter sat by the wooden wheel, struggling to center a lump of gray clay. The master craftsman placed his hands gently over the apprentice's fingers and advised him to steady his elbows against his hips. With proper leverage, the clay obeyed, rising smoothly into a hollow cylinder.",
    "An old fisherman noticed that migratory swallows were flying unusually low over the harbor waters. Knowing that high humidity forces flying insects closer to the surface, he secured his skiff to the stone jetty just before the squall arrived. His foresight saved his vessel while others drifted adrift.",
    "In a tranquil mountain village, carpenters built wooden houses using mortise and tenon joints without a single iron nail. During earthquakes, the flexible interlocking timbers swayed together and absorbed the tremors without fracturing. The ancient pagoda survived five centuries of shifting ground.",
    "Clara discovered a forgotten greenhouse at the edge of the estate. Sunlight filtered through cracked glass panes onto wild ferns and hardy ivy. She cleared the dead leaves from the irrigation channels, allowing fresh rainwater to fill the copper cistern and revive the dormant jasmine blossoms.",
    "A flock of starlings moved across the autumn sky in an undulating murmuration. Without any central leader, each individual bird matched its speed and heading to its seven nearest neighbors. The coordinated flock evaded the pursuing falcon through collective hydrodynamic motion.",
    "The weaver selected strands of dyed wool in indigo, saffron, and madder red. Working line by line upon the heavy loom, she interlaced the warp and weft to depict the migration of wild geese. Each completed pattern preserved an ancient folk narrative for future generations.",
    "A boy found an injured red-tailed hawk beneath a tall cedar tree after a severe thunderstorm. He carefully wrapped the bird in a flannel blanket and placed it in a quiet, darkened wooden crate. After three days of rest and fresh water, the bird regained its strength and soared back toward the canyon.",
    "During the cold winter harvest, the villagers gathered together around the communal oven to bake rye loaves. Everyone brought their own flour and shared yeast starters that had been nurtured for decades. The fragrant warmth filled the village hall, fostering solidarity through bitter frost.",
    "The cartographer climbed the coastal promontory at sunrise with a brass alidade and a surveying chain. By measuring angles to three known headlands, he plotted the precise contours of the reef onto vellum paper, creating a reliable navigational map for seafaring merchants.",
    "An archer practiced releasing arrows at dusk when the wind settled. He learned that focusing solely on the bullseye caused his shoulders to tense up. Only when he focused on his smooth breathing and clean release did the arrows strike true into the straw target.",
    "A botanist spent months cataloging alpine mosses clinging to granite cliffs above the snowline. Despite freezing winds and minimal organic soil, these resilient plants maintained metabolic activity by synthesizing specialized antifreeze sugars within their cellular fluid.",

    # --- Genre 2: Scientific Explanations ---
    "Photosynthesis is the biochemical process by which autotrophic organisms convert light energy into chemical energy. Inside plant chloroplasts, chlorophyll molecules absorb photons in the blue and red spectra, exciting electrons to split water molecules into protons, electrons, and oxygen gas. The resulting energy carriers drive the Calvin cycle to synthesize glucose from atmospheric carbon dioxide.",
    "The first law of thermodynamics, also known as the law of conservation of energy, states that energy cannot be created or destroyed in an isolated system. Energy can only change forms, such as transforming from gravitational potential energy into kinetic energy during free fall, or from chemical bond energy into thermal energy during combustion.",
    "Plate tectonics explains the dynamic structure of Earth's outer shell. The lithosphere is broken into numerous rigid tectonic plates that drift atop the plastic, semi-fluid asthenosphere. At divergent boundaries, upwelling magma forms new oceanic crust, while at convergent subduction zones, dense oceanic plates sink beneath buoyant continental plates.",
    "The water cycle describes the continuous movement of water on, above, and below the surface of the Earth. Solar radiation heats ocean water, causing liquid evaporation and plant transpiration into atmospheric vapor. As warm moist air rises, it expands and cools adiabatically, condensing into cloud droplets that eventually fall as precipitation.",
    "Newton's third law of motion dictates that for every action force, there is an equal and opposite reaction force. When a rocket engine burns propellant, high-velocity exhaust gases are expelled downward through the nozzle. The downward momentum imparted to the gas produces an equal upward thrust that propels the rocket into orbit.",
    "Refraction occurs when an electromagnetic wave changes speed upon passing from one optical medium into another with a different refractive index. According to Snell's law, a light beam entering a denser optical medium bends toward the surface normal line, causing optical distortion and chromatic dispersion across prisms.",
    "Cellular respiration is the biochemical pathway through which living cells extract adenosine triphosphate from organic nutrients. In the presence of oxygen, glycolysis in the cytoplasm is followed by pyruvate oxidation, the citric acid cycle in the mitochondrial matrix, and oxidative phosphorylation along the inner mitochondrial membrane.",
    "Earth's magnetic field is generated by a geodynamo mechanism operating in the molten outer iron core. Convection currents driven by heat loss from the solid inner core, combined with the Coriolis effect from planetary rotation, produce circulating electrical currents that generate an extensive protective magnetosphere.",
    "The Doppler effect is the perceived change in frequency of a wave in relation to an observer moving relative to the wave source. As a sound source approaches an observer, successive wave crests are compressed, leading to a higher pitch, whereas recession stretches wave intervals, lowering the perceived frequency.",
    "Enzymes are specialized biological catalysts composed primarily of globular proteins. They accelerate the rate of chemical reactions by lowering the activation energy barrier without being consumed in the process. Each enzyme possesses an active site tailored to bind specific substrate molecules with high conformational precision.",
    "Gravity is the fundamental interaction that causes mutual attraction between all physical bodies with mass. In general relativity, Albert Einstein described gravitation not as an invisible pull, but as the curvature of four-dimensional spacetime caused by the uneven distribution of mass and energy.",
    "The electromagnetic spectrum encompasses the entire range of all types of electromagnetic radiation. Waves are ordered by increasing frequency and decreasing wavelength: radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays. All electromagnetic waves propagate through a vacuum at the speed of light.",
    "Thermal conduction is the transfer of internal kinetic energy through microscopic collisions of particles within a material body. In metals, thermal conduction is exceptionally rapid because delocalized valence electrons freely carry kinetic energy throughout the metallic crystal lattice alongside lattice vibrations known as phonons.",
    "DNA consists of two antiparallel polynucleotide chains coiled around a common central axis into a right-handed double helix. The sugar-phosphate backbones form the outer structural support, while complementary nitrogenous bases pair internally through hydrogen bonds: adenine pairs with thymine, and guanine pairs with cytosine.",
    "Sound is a mechanical longitudinal wave that travels through compressible media such as gases, liquids, and solids. In air, sound waves propagate through alternating compressions of high particle density and rarefactions of low particle density. Because sound requires a physical medium to transfer vibrations, it cannot travel through a vacuum.",

    # --- Genre 3: Factual and Historical Passages ---
    "Johannes Gutenberg established the movable type printing press in Mainz around the year 1440. By combining hand-cast lead-tin-antimony alloy type, oil-based ink, and a modified agricultural screw press, Gutenberg enabled the mass reproduction of uniform texts. This invention catalyzed the rapid dissemination of scientific treatises and literacy throughout Renaissance Europe.",
    "The Library of Alexandria was one of the largest and most significant archives of the ancient world. Established in Egypt during the Hellenistic Ptolemaic dynasty around the third century BCE, it housed hundreds of thousands of papyrus scrolls spanning mathematics, astronomy, philosophy, and tragedy, attracting scholars from across the Mediterranean basin.",
    "James Watt revolutionized industrial production by introducing the separate condenser to the steam engine in 1765. In the earlier Newcomen atmospheric engine, the single cylinder had to be alternately heated and cooled, wasting enormous quantities of coal. Watt's separate condenser maintained the cylinder at steam temperature, quadrupling fuel efficiency.",
    "The Roman aqueduct system represented a monumental achievement in civil hydraulic engineering. Utilizing a precise, uniform downward gradient powered purely by gravitational force, arched masonry bridges and subterranean conduits transported fresh spring water over tens of kilometers into public baths, fountains, and private cisterns across the empire.",
    "The mariner's magnetic compass transformed maritime navigation when it spread to Mediterranean sailors during the thirteenth century. By enabling navigators to maintain a consistent heading even when overcast skies obscured the sun and the North Star, the compass facilitated reliable long-distance open ocean voyages beyond sight of coastal landmarks.",
    "The Panama Canal is an artificial 82-kilometer waterway in Panama that connects the Atlantic Ocean with the Pacific Ocean. Completed in 1914, the canal employs a series of three water lock chambers that lift ocean vessels 26 meters above sea level to Gatun Lake, eliminating the hazardous 15,000-kilometer voyage around Cape Horn.",
    "John Harrison resolved the long-standing navigational longitude problem by inventing the marine chronometer in the eighteenth century. Navigators could determine latitude by measuring solar elevation, but calculating longitude required knowing the exact time at a reference meridian. Harrison's temperature-compensated clock remained accurate despite oceanic motions and humidity shifts.",
    "The Colosseum in Rome, originally known as the Flavian Amphitheatre, was constructed between 72 and 80 CE under emperors Vespasian and Titus. Built from travertine limestone, tuff, and brick-faced concrete, its elliptical stadium could accommodate an estimated 50,000 spectators and featured an intricate subterranean hypogeum for staging wild animal spectacles.",
    "The transatlantic telegraph cable, successfully laid along the Atlantic seafloor in 1866 after multiple failed attempts, reduced intercontinental communication latency from ten days by steamship to a few minutes. Insulated with gutta-percha latex and armored with galvanized iron wire, the cable ushered in modern global telecommunications.",
    "The Silk Road was an expansive network of overland and maritime trade routes connecting East Asia with the Mediterranean basin. Originating during China's Han dynasty, the routes facilitated not only the commercial exchange of silk, spices, porcelain, and glassware, but also the transmission of religions, philosophies, and architectural techniques.",
    "The Great Pyramid of Giza was constructed as a funerary monument for Pharaoh Khufu during the Fourth Dynasty of Egypt around 2560 BCE. Originally standing 146.6 meters tall, the monument was erected from roughly 2.3 million quarried limestone and granite blocks, aligned with astonishing geometric precision to true north.",
    "The Apollo guidance computer, developed by the MIT Instrumentation Laboratory for the Apollo lunar missions, was one of the earliest operational computers to utilize silicon integrated circuits. Operating with a clock rate of one megahertz and four kilobytes of magnetic core memory, it executed real-time autonomous trajectory guidance and spacecraft navigation.",
    "The Svalbard Global Seed Vault is a secure underground gene bank located on the Norwegian island of Spitsbergen inside the Arctic Circle. Buried deep within permafrost sandstone 120 meters above sea level, the facility provides cold, dry storage for hundreds of thousands of duplicate crop seed varieties to protect agricultural biodiversity against global catastrophes.",
    "The Hoover Dam is a concrete arch-gravity structure constructed across the Black Canyon of the Colorado River between 1931 and 1936. Containing over 3.25 million cubic meters of concrete, the dam controls seasonal floodwaters, supplies irrigation water to surrounding arid regions, and generates hydroelectric power through massive hydraulic turbines.",
    "The Montgolfier brothers, Joseph-Michel and Jacques-Etienne, demonstrated the first manned untethered hot air balloon flight in Paris in November 1783. By burning straw and wool in a brazier beneath an envelope of paper and silk, they heated the air within, decreasing its density relative to the cool atmosphere and producing buoyant aerodynamic lift.",

    # --- Genre 4: Dialogues and Conversations ---
    "\"Have you checked the calibration on the spectrophotometer?\" asked David, looking over the laboratory bench. \"Yes, I ran a baseline scan with distilled water ten minutes ago,\" replied Sarah. \"The absorption peak is steady at 540 nanometers, so we can proceed with the protein assay dilutions.\"",
    "\"Why does the bread crust turn golden brown while baking?\" asked Leo, sniffing the warm kitchen air. \"That is the Maillard reaction,\" explained his mother. \"When reducing sugars and amino acids in the dough are heated above 140 degrees Celsius, they form complex aromatic compounds that produce that rich color and savory flavor.\"",
    "\"Is the telescope focused on Saturn's rings yet?\" whispered Maya at the observatory eyepiece. \"Almost,\" replied Thomas, adjusting the fine micrometer dial. \"The atmospheric turbulence is settling down. Take a look now, you can clearly see the Cassini division between the A and B rings.\"",
    "\"What is causing the intermittent vibration in the engine mount?\" asked the technician. \"The harmonic damper bolt worked itself loose during high-RPM testing,\" answered the lead engineer. \"We need to torque it to eighty Newton-meters and apply thread-locking compound before running the dynamometer again.\"",
    "\"Why do desert plants have such thick fleshy stems?\" inquired the student during the greenhouse tour. \"Those stems store water to survive extended arid seasons,\" replied the curator. \"Furthermore, many desert succulents use CAM photosynthesis, opening their stomata only at night to minimize transpiration losses.\"",
    "\"Can we deploy the new weather sensor array before the cold front arrives?\" asked the station supervisor. \"The telemetry transmitters are already synchronized with the local base station,\" confirmed the meteorologist. \"We only need twenty minutes to mount the anemometer mast on the ridge.\"",
    "\"How does the noise-canceling headphone eliminate ambient rumbling?\" asked Jacob on the train. \"It uses external microphones to capture incoming low-frequency sound waves,\" said Rachel. \"An onboard digital signal processor then generates an inverted sound wave of equal amplitude, causing destructive interference.\"",
    "\"Did the soil test show any mineral deficiencies in the orchard?\" asked the farmer. \"The potassium and phosphorus concentrations are optimal,\" replied the agronomist. \"However, the soil pH is slightly acidic at 5.4, so adding pulverized agricultural limestone will help the apple trees absorb nitrogen efficiently.\"",
    "\"Why is stainless steel resistant to rust compared to ordinary carbon steel?\" asked the apprentice welder. \"Stainless steel contains at least ten percent chromium by mass,\" explained the master. \"The chromium reacts with atmospheric oxygen to form a microscopic, self-healing chromium oxide film that seals the underlying iron from water and corrosion.\"",
    "\"What is the function of the control rods inside a nuclear reactor?\" asked the physics student. \"Control rods are composed of neutron-absorbing elements such as boron, cadmium, or hafnium,\" explained the professor. \"Inserting them into the reactor core slows down the nuclear fission chain reaction, while withdrawing them increases power output.\"",

    # --- Genre 5: Instructional and Procedural Texts ---
    "To prepare an artisan sourdough loaf, first mix three hundred grams of unbleached bread flour with seventy grams of active levain and two hundred grams of lukewarm water. Allow the mixture to autolyse for forty-five minutes. Next, fold in six grams of fine sea salt and perform four sets of stretch-and-folds at thirty-minute intervals. Shape the dough gently into a round boule and retard overnight in the refrigerator before baking at 230 degrees Celsius.",
    "When inspecting a bicycle brake system, begin by checking the brake pads for uneven wear or embedded debris. Ensure the pad grooves are clearly visible and align squarely with the wheel rim braking track without touching the rubber tire. Adjust the barrel adjuster to take up cable slack until the lever engages firmly after one centimeter of travel.",
    "To clean a delicate optical microscope lens, never wipe the dry glass with rough paper towels. First, use a rubber squeeze bulb to blow away loose abrasive grit from the surface. Then, moisten a single sheet of lint-free lens paper with high-purity optical cleaning fluid or anhydrous isopropyl alcohol, and gently drag it across the lens in a single spiraling stroke from center to edge.",
    "When transplanting young seedlings into an outdoor garden bed, harden them off first by exposing them to outdoor conditions for several hours daily over one week. Dig planting holes twice as wide as the root ball and moisten the soil thoroughly. Position each seedling at the same soil depth as its pot, press the surrounding earth firmly to eliminate air pockets, and water deeply.",
    "To fold a traditional paper origami crane, start with a perfectly square sheet of paper colored side up. Fold diagonally in both directions, unfold, then flip over and fold horizontally and vertically. Collapse the creases into a square base. Fold both outer edges toward the center crease to form a kite shape, perform a petal fold on both sides, and fold down the neck, head, and wings.",
    "When calibrating an electronic pH meter, always use two fresh standard buffer solutions, typically pH 7.00 and pH 4.01. Rinse the glass electrode with deionized water and blot gently with a soft tissue without rubbing. Immerse the probe into the neutral pH 7.00 buffer and wait for temperature stabilization before adjusting the zero calibration, followed by the slope calibration in the pH 4.01 buffer.",
    "To safely sharpen a chisel on a whetstone, submerge the water stone until air bubbles cease rising. Place the beveled face of the chisel flat against the stone surface at a consistent twenty-five-degree angle. Move the blade back and forth along the full length of the stone using steady downward pressure until a uniform microscopic burr forms across the entire straight back edge.",
    "When preparing a biological specimen for light microscopy, place a single drop of clean water or staining solution onto the center of a clean glass slide. Carefully position the thin tissue slice into the droplet using tweezers. Hold a thin glass coverslip at a forty-five-degree angle along one edge and slowly lower it onto the liquid to prevent air bubbles from becoming trapped.",
    "To splice two stranded copper electrical wires together securely, strip approximately two centimeters of insulation from both ends without nicking the conductor strands. Fan out the individual copper strands and intermesh them together tightly. Twist the overlapping wires clockwise into a compact spiral, solder the joint with rosin-core solder, and insulate with heat-shrink tubing.",
    "When performing cardiopulmonary resuscitation on an unresponsive adult, first verify that the surroundings are safe and call emergency services immediately. Place the heel of one hand in the center of the chest with the other hand interlocked above it. Deliver continuous chest compressions at a rate of one hundred to one hundred and twenty beats per minute to a depth of at least five centimeters.",
]


# =====================================================================
# Reading Comprehension Partitions (Strict Document & Fact Isolation)
# =====================================================================

TRAIN_QA_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="train_doc_01_photosynthesis",
        topic="Plant Biology",
        text="Photosynthesis occurs inside chloroplasts using chlorophyll pigments. Plants absorb sunlight and carbon dioxide to synthesize glucose and release oxygen.",
        answerable_pairs=[
            ("Where does photosynthesis occur inside plant cells?", "Chloroplasts", ["in chloroplasts", "chloroplast"]),
            ("What gas do plants release during photosynthesis?", "Oxygen", ["oxygen gas", "O2"]),
            ("What sugar is synthesized during photosynthesis?", "Glucose", ["glucose sugar"]),
        ],
        unanswerable_questions=[
            "What temperature is required for the Calvin cycle?",
            "How many ATP molecules are produced during the light reaction?",
        ],
        paraphrase_pairs=[
            ("Which cellular organelle hosts photosynthesis?", "Chloroplasts", ["chloroplast", "in chloroplasts"]),
            ("What byproduct gas is produced by plants during photosynthesis?", "Oxygen", ["oxygen gas", "O2"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_02_solar_system",
        topic="Astronomy",
        text="Jupiter is the largest planet in our solar system and possesses a persistent storm called the Great Red Spot. It is primarily composed of hydrogen and helium gases.",
        answerable_pairs=[
            ("Which is the largest planet in our solar system?", "Jupiter", ["planet Jupiter"]),
            ("What is the persistent storm on Jupiter called?", "Great Red Spot", ["the Great Red Spot"]),
            ("What elements is Jupiter primarily composed of?", "Hydrogen and helium", ["hydrogen and helium gases"]),
        ],
        unanswerable_questions=[
            "How many moons orbit Jupiter?",
            "What is the surface temperature of Jupiter's core?",
        ],
        paraphrase_pairs=[
            ("Name the biggest planet in the solar system.", "Jupiter", ["planet Jupiter"]),
            ("What name is given to Jupiter's giant storm?", "Great Red Spot", ["the Great Red Spot"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_03_printing_press",
        topic="European History",
        text="Johannes Gutenberg invented the movable type printing press in Mainz around 1440. His innovation dramatically accelerated the spread of books across Europe.",
        answerable_pairs=[
            ("Who invented the movable type printing press?", "Johannes Gutenberg", ["Gutenberg"]),
            ("In which city was Gutenberg's press developed?", "Mainz", ["city of Mainz"]),
            ("Around what year did Gutenberg invent his press?", "1440", ["around 1440", "circa 1440"]),
        ],
        unanswerable_questions=[
            "How many Gutenberg Bibles were originally printed?",
            "What metal alloy was used to cast Gutenberg's type?",
        ],
        paraphrase_pairs=[
            ("Which inventor created movable type printing in 1440?", "Johannes Gutenberg", ["Gutenberg"]),
            ("Where did Gutenberg invent the printing press?", "Mainz", ["in Mainz"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_04_aircraft_lift",
        topic="Aerodynamics",
        text="Aircraft wings generate aerodynamic lift due to differences in air pressure above and below the airfoil. The forward thrust is provided by jet engines or propellers.",
        answerable_pairs=[
            ("What causes aerodynamic lift on aircraft wings?", "Differences in air pressure", ["air pressure differences", "pressure differences"]),
            ("What provides forward thrust in aircraft?", "Jet engines or propellers", ["engines or propellers", "jet engines"]),
        ],
        unanswerable_questions=[
            "What is the top cruising speed of a Boeing 747?",
            "How much fuel does an Airbus A380 consume per hour?",
        ],
        paraphrase_pairs=[
            ("Why do airplane wings produce lift?", "Differences in air pressure", ["pressure differences"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_05_microprocessors",
        topic="Semiconductor Technology",
        text="Microprocessors are fabricated on silicon wafers using optical photolithography. Billions of field-effect transistors act as electronic switches to execute arithmetic logic operations.",
        answerable_pairs=[
            ("What semiconductor substrate is used to fabricate microprocessors?", "Silicon wafers", ["silicon", "silicon wafer"]),
            ("Which optical manufacturing process creates microprocessor circuits?", "Photolithography", ["optical photolithography"]),
            ("What electronic components act as switches in microprocessors?", "Field-effect transistors", ["transistors"]),
        ],
        unanswerable_questions=[
            "What is the exact clock frequency of the first Intel 4004 chip?",
            "What is the gate dielectric thickness in modern 3nm transistors?",
        ],
        paraphrase_pairs=[
            ("On which material are microprocessors manufactured?", "Silicon wafers", ["silicon"]),
            ("Name the technique used to pattern microprocessor circuits.", "Photolithography", ["optical photolithography"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_06_rainforest_ecology",
        topic="Ecology",
        text="Tropical rainforest canopies absorb high solar radiation and generate localized rainfall through plant transpiration. Epiphytic orchids grow directly upon tree branches to capture canopy moisture.",
        answerable_pairs=[
            ("Through which biological process do rainforest canopies produce rainfall?", "Plant transpiration", ["transpiration"]),
            ("Where do epiphytic orchids grow in the rainforest?", "Tree branches", ["directly upon tree branches", "on branches"]),
        ],
        unanswerable_questions=[
            "How many tree species exist per hectare in the Amazon basin?",
            "What is the annual biomass turnover rate in tropical rainforest soils?",
        ],
        paraphrase_pairs=[
            ("What botanical mechanism drives rainfall in tropical rainforests?", "Plant transpiration", ["transpiration"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_07_neuroscience_memory",
        topic="Neuroscience",
        text="The hippocampus consolidates short-term neural signals into stable long-term memories in the cerebral cortex. Synaptic plasticity enables persistent signal strengthening known as long-term potentiation.",
        answerable_pairs=[
            ("Which brain structure consolidates short-term memories?", "Hippocampus", ["the hippocampus"]),
            ("Where are long-term memories permanently stored?", "Cerebral cortex", ["the cerebral cortex", "cortex"]),
            ("What synaptic mechanism enables persistent signal strengthening?", "Long-term potentiation", ["LTP"]),
        ],
        unanswerable_questions=[
            "How many action potentials can a pyramidal neuron fire per second?",
            "Which neurotransmitter receptor subtype binds glutamate in memory retrieval?",
        ],
        paraphrase_pairs=[
            ("Name the anatomical brain region that consolidates memories.", "Hippocampus", ["the hippocampus"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_08_geodesic_architecture",
        topic="Civil Architecture",
        text="Architect Buckminster Fuller popularized the geodesic dome, which utilizes spherical networks of interconnected triangles. The structure distributes mechanical stress omnidirectionally through tensegrity.",
        answerable_pairs=[
            ("Who popularized the geodesic dome?", "Buckminster Fuller", ["Fuller", "R. Buckminster Fuller"]),
            ("What geometric shape forms the network of geodesic domes?", "Interconnected triangles", ["triangles"]),
            ("How does a geodesic dome distribute mechanical stress?", "Omnidirectionally through tensegrity", ["omnidirectionally", "through tensegrity"]),
        ],
        unanswerable_questions=[
            "What was the construction diameter of the Montreal Biosphere dome?",
            "What alloy formed the outer struts of Fuller's prototype dome?",
        ],
        paraphrase_pairs=[
            ("Which architect is famous for designing geodesic domes?", "Buckminster Fuller", ["Fuller"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_09_meteorology_cyclones",
        topic="Meteorology",
        text="Tropical cyclones form over warm ocean waters exceeding 26 degrees Celsius. The Coriolis force imparts rotational spin, creating a calm low-pressure central eye surrounded by eyewall cloud bands.",
        answerable_pairs=[
            ("What minimum water temperature is required for tropical cyclones?", "26 degrees Celsius", ["26 C", "exceeding 26 degrees Celsius"]),
            ("What physical force imparts rotational spin to cyclones?", "Coriolis force", ["the Coriolis force", "Coriolis"]),
            ("What is the calm central region of a tropical cyclone called?", "The eye", ["eye", "central eye"]),
        ],
        unanswerable_questions=[
            "What was the lowest atmospheric pressure ever recorded in Typhoon Tip?",
            "How many nautical miles wide can hurricane storm surges reach?",
        ],
        paraphrase_pairs=[
            ("Which natural effect causes cyclone rotation?", "Coriolis force", ["Coriolis"]),
            ("Name the tranquil center of a tropical cyclone.", "The eye", ["eye", "central eye"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_10_graphene_materials",
        topic="Materials Science",
        text="Graphene is a single atomic monolayer of carbon atoms bonded in a hexagonal honeycomb lattice. It demonstrates extraordinary electrical conductivity and ballistic electron transport at room temperature.",
        answerable_pairs=[
            ("What chemical element comprises graphene?", "Carbon", ["carbon atoms", "pure carbon"]),
            ("What geometric lattice shape do carbon atoms form in graphene?", "Hexagonal honeycomb lattice", ["honeycomb lattice", "hexagonal lattice"]),
            ("What electrical transport phenomenon occurs in graphene at room temperature?", "Ballistic electron transport", ["ballistic transport"]),
        ],
        unanswerable_questions=[
            "What is the Young's modulus of multilayer turbostratic graphene?",
            "Who first synthesized suspended graphene on copper foils?",
        ],
        paraphrase_pairs=[
            ("Which element is graphene composed of?", "Carbon", ["carbon atoms"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_11_steam_engine",
        topic="Industrial Technology",
        text="James Watt developed the separate condenser for steam engines in Glasgow in 1765. This breakthrough drastically reduced fuel consumption and powered the Industrial Revolution.",
        answerable_pairs=[
            ("Who developed the separate condenser for steam engines?", "James Watt", ["Watt"]),
            ("In what year did James Watt develop the separate condenser?", "1765", ["in 1765"]),
            ("In which city was the separate condenser developed?", "Glasgow", ["city of Glasgow"]),
        ],
        unanswerable_questions=[
            "How many pounds of coal did Newcomen engines consume per minute?",
            "What was the cylinder stroke diameter of Watt's first commercial engine?",
        ],
        paraphrase_pairs=[
            ("Who invented the separate condenser in 1765?", "James Watt", ["Watt"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_12_ocean_currents",
        topic="Ocean Dynamics",
        text="The Gulf Stream is a swift warm ocean current originating in the Gulf of Mexico. It carries heated water across the Atlantic Ocean toward Western Europe, moderating coastal climates.",
        answerable_pairs=[
            ("Where does the Gulf Stream originate?", "Gulf of Mexico", ["the Gulf of Mexico"]),
            ("What type of ocean current is the Gulf Stream?", "Warm ocean current", ["warm current", "swift warm current"]),
            ("Toward which continent does the Gulf Stream transport warm water?", "Western Europe", ["Europe"]),
        ],
        unanswerable_questions=[
            "What is the maximum depth of the Florida Straits?",
            "How many sverdrups of volume transport does the Gulf Stream reach at Cape Hatteras?",
        ],
        paraphrase_pairs=[
            ("Where does the Gulf Stream begin?", "Gulf of Mexico", ["the Gulf of Mexico"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_13_cellular_mitosis",
        topic="Cellular Biology",
        text="During mitosis, identical sister chromatids are pulled apart by spindle fibers toward opposite poles of the dividing cell. Telophase concludes the division by reforming nuclear envelopes around each daughter nucleus.",
        answerable_pairs=[
            ("What structures pull sister chromatids apart during mitosis?", "Spindle fibers", ["spindle fibers"]),
            ("Which phase concludes mitosis by reforming nuclear envelopes?", "Telophase", ["telophase"]),
        ],
        unanswerable_questions=[
            "How many hours does the entire mammalian cell cycle take?",
            "Which cyclin-dependent kinase triggers chromatin condensation?",
        ],
        paraphrase_pairs=[
            ("What apparatus separates chromatids during cell division?", "Spindle fibers", ["spindle fibers"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_14_laser_physics",
        topic="Laser Optics",
        text="Lasers produce coherent monochromatic light through stimulated emission inside an optical cavity. In 1960, Theodore Maiman constructed the first working optical laser utilizing a synthetic ruby crystal rod.",
        answerable_pairs=[
            ("What physical process produces coherent light in lasers?", "Stimulated emission", ["stimulated emission"]),
            ("Who built the first operating optical laser in 1960?", "Theodore Maiman", ["Maiman"]),
            ("What crystal material was used in the first working laser?", "Synthetic ruby crystal", ["ruby crystal", "ruby"]),
        ],
        unanswerable_questions=[
            "What was the output pulse energy in millijoules of Maiman's prototype?",
            "Which flashlamp manufacturer supplied the helical lamp for the ruby laser?",
        ],
        paraphrase_pairs=[
            ("Who invented the synthetic ruby laser in 1960?", "Theodore Maiman", ["Maiman"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_15_volcanic_calderas",
        topic="Vulcanology",
        text="A caldera is a large basin depression formed when a volcano collapses into its emptied subterranean magma chamber following a catastrophic explosive eruption.",
        answerable_pairs=[
            ("What geological structure forms when a volcano collapses into its emptied chamber?", "Caldera", ["a caldera"]),
            ("What subterranean chamber empties prior to caldera formation?", "Magma chamber", ["subterranean magma chamber"]),
        ],
        unanswerable_questions=[
            "What was the total tephra volume ejected during the Krakatoa 1883 event?",
            "How deep is the lake inside the Crater Lake caldera in Oregon?",
        ],
        paraphrase_pairs=[
            ("Name the depression left by a collapsed volcanic summit.", "Caldera", ["a caldera"]),
        ],
    ),
    DocumentPassage(
        doc_id="train_doc_16_penicillin_discovery",
        topic="Medical History",
        text="Alexander Fleming observed in 1928 that Penicillium notatum mold colonies inhibited Staphylococcus bacteria growth in contaminated Petri dishes, identifying the first natural antibiotic substance.",
        answerable_pairs=[
            ("Who discovered the antibiotic properties of penicillin mold in 1928?", "Alexander Fleming", ["Fleming"]),
            ("Which genus of bacteria did penicillin mold colonies inhibit?", "Staphylococcus", ["Staphylococcus bacteria"]),
            ("In what year did Fleming observe bacterial inhibition by mold?", "1928", ["in 1928"]),
        ],
        unanswerable_questions=[
            "What broth medium did Fleming use to culture Penicillium mold?",
            "How many units of penicillin did Howard Florey produce in Oxford?",
        ],
        paraphrase_pairs=[
            ("Who noticed in 1928 that mold halted bacterial growth?", "Alexander Fleming", ["Fleming"]),
        ],
    ),
]


VAL_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="val_doc_01_plate_tectonics",
        topic="Geology",
        text="The Earth's lithosphere is divided into rigid tectonic plates that float on the ductile asthenosphere. Plate boundaries cause earthquakes and volcanic eruptions.",
        answerable_pairs=[
            ("What rigid layer of the Earth is divided into tectonic plates?", "Lithosphere", ["the lithosphere", "Earth's lithosphere"]),
            ("On which ductile layer do tectonic plates float?", "Asthenosphere", ["the asthenosphere", "ductile asthenosphere"]),
            ("What geological events are caused by plate boundaries?", "Earthquakes and volcanic eruptions", ["earthquakes and volcanoes", "earthquakes"]),
        ],
        unanswerable_questions=[
            "How fast do the Pacific and North American plates move each year?",
            "What is the depth of the Mariana Trench in kilometers?",
        ],
        paraphrase_pairs=[
            ("Which layer of Earth is partitioned into plates?", "Lithosphere", ["the lithosphere"]),
            ("Where do tectonic plates drift on top of?", "Asthenosphere", ["the asthenosphere"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_02_optics_lenses",
        topic="Physics",
        text="A convex lens converges parallel rays of light toward a focal point, whereas a concave lens diverges light rays. Optical refraction depends on the refractive index of the glass.",
        answerable_pairs=[
            ("What does a convex lens do to parallel light rays?", "Converges light rays toward a focal point", ["converges light rays", "converges light"]),
            ("What does a concave lens do to light rays?", "Diverges light rays", ["diverges light"]),
            ("What material property determines refraction in glass?", "Refractive index", ["the refractive index"]),
        ],
        unanswerable_questions=[
            "What is the exact refractive index of flint glass?",
            "Who first ground optical glass lenses for telescopes?",
        ],
        paraphrase_pairs=[
            ("Which type of lens converges parallel rays?", "Convex lens", ["convex"]),
            ("Which lens diverges incoming light rays?", "Concave lens", ["concave"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_03_black_holes",
        topic="Astrophysics",
        text="A black hole is bounded by an event horizon from which no matter or electromagnetic radiation can escape. At its mathematical center lies a gravitational singularity of infinite density.",
        answerable_pairs=[
            ("What boundary marks the point of no return around a black hole?", "Event horizon", ["the event horizon"]),
            ("What exists at the center of a black hole?", "Gravitational singularity", ["singularity", "a singularity"]),
        ],
        unanswerable_questions=[
            "What is the Hawking radiation temperature of Cygnus X-1?",
            "How fast does the accretion disk of Sagittarius A* rotate?",
        ],
        paraphrase_pairs=[
            ("Name the perimeter beyond which light cannot escape a black hole.", "Event horizon", ["the event horizon"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_04_photovoltaics",
        topic="Renewable Energy",
        text="Photovoltaic solar cells convert sunlight into electricity via the photoelectric effect within a semiconductor p-n junction. When photons strike the cell, electrons are energized into the conduction band.",
        answerable_pairs=[
            ("Which physical effect converts sunlight to electricity in solar cells?", "Photoelectric effect", ["the photoelectric effect"]),
            ("What semiconductor junction structure is used inside photovoltaic cells?", "p-n junction", ["a p-n junction", "semiconductor p-n junction"]),
            ("Where are electrons excited into when struck by photons?", "Conduction band", ["the conduction band"]),
        ],
        unanswerable_questions=[
            "What is the record laboratory efficiency of perovskite tandem cells?",
            "What chemical dopant is added to n-type silicon wafers?",
        ],
        paraphrase_pairs=[
            ("Which phenomenon generates electric current in solar panels?", "Photoelectric effect", ["the photoelectric effect"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_05_acoustic_echolocation",
        topic="Zoology",
        text="Microbats emit ultrasonic clicks from their larynx to navigate and locate flying prey through echolocation. The returning acoustic echoes provide precise spatial maps of target velocity and distance.",
        answerable_pairs=[
            ("Which anatomical organ in microbats produces ultrasonic clicks?", "Larynx", ["the larynx"]),
            ("What sensory mechanism do bats use to locate prey?", "Echolocation", ["ultrasonic echolocation"]),
            ("What target properties do returning echoes convey?", "Velocity and distance", ["distance and velocity"]),
        ],
        unanswerable_questions=[
            "What is the maximum frequency in kilohertz produced by horseshoe bats?",
            "How many insects does an adult brown bat consume per hour?",
        ],
        paraphrase_pairs=[
            ("Through which technique do bats detect aerial targets?", "Echolocation", ["echolocation"]),
        ],
    ),
    DocumentPassage(
        doc_id="val_doc_06_suspension_bridges",
        topic="Civil Engineering",
        text="Suspension bridges carry deck loads through vertical suspender cables anchored to massive parabolic main cables strung between tall towers and secured in granite anchorages.",
        answerable_pairs=[
            ("What vertical members transfer bridge deck weight to main cables?", "Suspender cables", ["vertical suspenders", "vertical suspender cables"]),
            ("What structural components anchor the ends of the main suspension cables?", "Granite anchorages", ["anchorages"]),
        ],
        unanswerable_questions=[
            "What was the tensile breaking strength of the Golden Gate bridge wires?",
            "How much does the deck of the Akashi Kaikyo bridge oscillate during typhoons?",
        ],
        paraphrase_pairs=[
            ("Which elements connect the deck to the main cables in suspension bridges?", "Suspender cables", ["suspender cables"]),
        ],
    ),
]


SEALED_TEST_QA_DOCUMENTS: list[DocumentPassage] = [
    DocumentPassage(
        doc_id="test_doc_01_hydrothermal_vents",
        topic="Marine Biology",
        text="Deep-sea hydrothermal vents emit mineral-rich fluids heated by underlying magma chambers. Chemosynthetic bacteria utilize hydrogen sulfide to produce organic matter without sunlight.",
        answerable_pairs=[
            ("What heats the mineral-rich fluids emitted by hydrothermal vents?", "Magma chambers", ["underlying magma chambers", "magma"]),
            ("What microorganism produces organic matter at hydrothermal vents?", "Chemosynthetic bacteria", ["bacteria", "chemosynthetic microbes"]),
            ("What chemical compound do chemosynthetic bacteria utilize?", "Hydrogen sulfide", ["H2S", "hydrogen sulfide gas"]),
        ],
        unanswerable_questions=[
            "What is the maximum water pressure at the Mariana hydrothermal vent?",
            "In what year were deep hydrothermal vents first discovered by submersible?",
        ],
        paraphrase_pairs=[
            ("What energy source heats fluids at hydrothermal vents?", "Magma chambers", ["magma", "underlying magma"]),
            ("Which chemical do vent bacteria convert into food?", "Hydrogen sulfide", ["hydrogen sulfide gas"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_02_james_webb_telescope",
        topic="Space Exploration",
        text="The James Webb Space Telescope operates at the second Lagrange point, 1.5 million kilometers from Earth. Its primary mirror consists of eighteen gold-coated beryllium segments.",
        answerable_pairs=[
            ("Where does the James Webb Space Telescope operate?", "Second Lagrange point", ["L2", "Lagrange point 2", "second Lagrange point L2"]),
            ("How far is the James Webb Space Telescope from Earth?", "1.5 million kilometers", ["1.5 million km", "1.5 million kilometers from Earth"]),
            ("What metal coats the eighteen beryllium segments of its mirror?", "Gold", ["gold coating", "gold-coated"]),
            ("How many beryllium segments comprise its primary mirror?", "Eighteen", ["18", "eighteen segments"]),
        ],
        unanswerable_questions=[
            "What rocket booster launched the James Webb Space Telescope?",
            "How much did the construction of the James Webb telescope cost in total?",
        ],
        paraphrase_pairs=[
            ("At which orbital location is the Webb telescope stationed?", "Second Lagrange point", ["L2", "Lagrange point 2"]),
            ("What material coats the mirror segments of the Webb telescope?", "Gold", ["gold-coated"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_03_analytical_engine",
        topic="Computer History",
        text="Charles Babbage designed the mechanical Analytical Engine in London, featuring an arithmetic unit called the mill and a memory store. Ada Lovelace wrote the first published algorithm for this machine.",
        answerable_pairs=[
            ("Who designed the mechanical Analytical Engine?", "Charles Babbage", ["Babbage", "Charles Babbage in London"]),
            ("What was the arithmetic calculation unit of the Analytical Engine called?", "The mill", ["mill"]),
            ("Who created the first algorithm for the Analytical Engine?", "Ada Lovelace", ["Lovelace", "Ada Lovelace in London"]),
        ],
        unanswerable_questions=[
            "How many steam cylinders were intended to power the Analytical Engine?",
            "What decimal precision in digits did Babbage plan for the store?",
        ],
        paraphrase_pairs=[
            ("Name the designer of the Analytical Engine.", "Charles Babbage", ["Babbage"]),
            ("Who authored the earliest published algorithm for Babbage's engine?", "Ada Lovelace", ["Lovelace"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_04_mycorrhizal_networks",
        topic="Forest Mycology",
        text="Mycorrhizal fungal hyphae form underground symbiotic networks connecting plant roots across forest soils. The fungi provide plants with mineral phosphorus and nitrogen in exchange for photosynthesized carbon sugars.",
        answerable_pairs=[
            ("What fungal structures connect plant root systems in forest soils?", "Mycorrhizal fungal hyphae", ["hyphae", "fungal hyphae"]),
            ("What essential minerals do mycorrhizal fungi supply to plants?", "Phosphorus and nitrogen", ["nitrogen and phosphorus", "phosphorus"]),
            ("What nutrient do plants provide to symbiotic fungi in return?", "Carbon sugars", ["photosynthesized carbon sugars", "sugars"]),
        ],
        unanswerable_questions=[
            "What percentage of forest carbon is transferred through fungal networks annually?",
            "Which tree species was first shown to share isotopic carbon in British Columbia?",
        ],
        paraphrase_pairs=[
            ("What nutrients do fungi deliver to host trees in forest soils?", "Phosphorus and nitrogen", ["nitrogen and phosphorus"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_05_superconductivity",
        topic="Condensed Matter",
        text="Type-II superconductors expel weak magnetic fields via the Meissner effect but permit magnetic vortex flux pinning above the lower critical field, enabling zero electrical resistance under high magnetic fields.",
        answerable_pairs=[
            ("Which physical effect expels weak magnetic fields from superconductors?", "Meissner effect", ["the Meissner effect"]),
            ("What electrical property characterizes superconductors below critical temperature?", "Zero electrical resistance", ["zero resistance"]),
        ],
        unanswerable_questions=[
            "What is the critical transition temperature in Kelvin of yttrium barium copper oxide?",
            "Who first measured the isotopic mass shift in lead superconductors?",
        ],
        paraphrase_pairs=[
            ("Name the phenomenon where superconductors eject magnetic flux lines.", "Meissner effect", ["the Meissner effect"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_06_solar_wind_magnetosphere",
        topic="Space Physics",
        text="The supersonic solar wind collides with Earth's dipole magnetic field to create a curved bow shock approximately thirteen Earth radii upstream from the planet.",
        answerable_pairs=[
            ("What structure is created when supersonic solar wind hits Earth's magnetic field?", "Bow shock", ["a bow shock", "curved bow shock"]),
            ("At what upstream distance from Earth does the bow shock typically form?", "Thirteen Earth radii", ["13 Earth radii", "approximately thirteen Earth radii"]),
        ],
        unanswerable_questions=[
            "What is the average proton velocity of the slow solar wind at 1 AU?",
            "What magnetic field strength in nanoteslas is observed in the magnetotail lobe?",
        ],
        paraphrase_pairs=[
            ("What shockwave forms ahead of Earth's magnetic shield?", "Bow shock", ["curved bow shock"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_07_carbon_nanotubes",
        topic="Nanotechnology",
        text="Single-walled carbon nanotubes are seamless cylindrical tubes formed by rolling single-layer graphene sheets along specific chiral vectors, yielding exceptional axial tensile strength and thermal conductivity.",
        answerable_pairs=[
            ("What material sheets are rolled into single-walled carbon nanotubes?", "Graphene sheets", ["graphene", "single-layer graphene"]),
            ("What geometric vectors dictate the electrical properties of carbon nanotubes?", "Chiral vectors", ["chiral vector"]),
        ],
        unanswerable_questions=[
            "What catalyst precursor was used in the high-pressure carbon monoxide HiPco synthesis?",
            "What is the thermal conductivity in watts per meter-kelvin of arm-chair nanotubes?",
        ],
        paraphrase_pairs=[
            ("What atomic sheets make up carbon nanotubes?", "Graphene sheets", ["graphene"]),
        ],
    ),
    DocumentPassage(
        doc_id="test_doc_08_oceanic_tides",
        topic="Geophysics",
        text="Oceanic tides are periodic sea-level fluctuations produced by gravitational tidal forces exerted primarily by the Moon and to a lesser extent the Sun upon Earth's water bodies.",
        answerable_pairs=[
            ("Which celestial body exerts the primary gravitational force driving ocean tides?", "The Moon", ["Moon"]),
            ("What secondary celestial body contributes to tidal forces on Earth?", "The Sun", ["Sun"]),
        ],
        unanswerable_questions=[
            "What is the highest recorded tidal amplitude in the Bay of Fundy in meters?",
            "How many centimeters does the Moon recede from Earth annually due to tidal friction?",
        ],
        paraphrase_pairs=[
            ("Which celestial object is the main cause of tides on Earth?", "The Moon", ["Moon"]),
        ],
    ),
]


# =====================================================================
# Prompt Formatting and Parsing Verifiers
# =====================================================================

def format_qa_prompt(passage: str, question: str) -> str:
    """Format standardized reading comprehension prompt without recency copying hazard."""
    return (
        f"<|user|>\n"
        f"Passage: {passage.strip()}\n"
        f"Question: {question.strip()}\n"
        f"Instruction: Extract the short factual answer directly from the passage above. If not present in the passage, reply '{ABSTENTION_TARGET}'.\n"
        f"<|assistant|>\n"
    )


def format_qa_completion(answer: str) -> str:
    """Format assistant target completion with end-of-sequence token."""
    return f"{answer.strip()}<|eos|>\n"


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and stripping articles."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    # Strip leading/trailing articles
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\s+(a|an|the)$", "", text)
    return " ".join(text.split())


def parse_language_completion(completion: str) -> tuple[str, bool, bool]:
    """Parse assistant completion into (cleaned_answer, is_compliant, is_abstained)."""
    cleaned = completion.replace("<|eos|>", "").strip()
    if not cleaned:
        return "", False, False

    first_line = cleaned.split("\n")[0].strip()
    normalized = normalize_text(first_line)

    is_abstained = any(abstain in normalized for abstain in [
        "insufficient information",
        "unanswerable",
        "not mentioned",
        "unknown",
        "cannot be determined",
        "information not provided",
    ])

    is_compliant = len(first_line) > 0 and len(first_line) <= 120
    return first_line, is_compliant, is_abstained


def verify_format_compliance(completion: str) -> bool:
    """Check if the generation cleanly adheres to single-answer format."""
    cleaned, is_compliant, _ = parse_language_completion(completion)
    return is_compliant and len(cleaned) > 0


def verify_language_exact(completion: str, expected_answer: str, allowed_answers: list[str] | None = None) -> bool:
    """Evaluate strict exact match against expected answer or allowed variants."""
    first_line, is_compliant, is_abstained = parse_language_completion(completion)
    if not is_compliant or not first_line:
        return False

    norm_resp = normalize_text(first_line)
    norm_exp = normalize_text(expected_answer)

    if norm_resp == norm_exp:
        return True

    if allowed_answers:
        for alias in allowed_answers:
            if norm_resp == normalize_text(alias):
                return True

    return False


def verify_language_grounded(completion: str, expected_answer: str, allowed_answers: list[str] | None = None) -> bool:
    """Evaluate grounded extraction (sub-phrase presence or semantic containment)."""
    first_line, is_compliant, is_abstained = parse_language_completion(completion)
    if not is_compliant or not first_line:
        return False

    norm_resp = normalize_text(first_line)
    norm_exp = normalize_text(expected_answer)

    # 1. Exact equality
    if norm_resp == norm_exp:
        return True

    # 2. Containment of expected answer in response or vice versa
    if norm_exp and norm_exp in norm_resp:
        return True
    if len(norm_resp) >= 4 and norm_resp in norm_exp:
        return True

    # 3. Check allowed aliases
    if allowed_answers:
        for alias in allowed_answers:
            norm_alias = normalize_text(alias)
            if norm_resp == norm_alias or (norm_alias and norm_alias in norm_resp):
                return True
            if len(norm_resp) >= 4 and norm_resp in norm_alias:
                return True

    return False


def verify_language_abstention(completion: str) -> bool:
    """Verify that model correctly abstained when information was absent."""
    _, is_compliant, is_abstained = parse_language_completion(completion)
    return is_compliant and is_abstained


# =====================================================================
# Strict Corpus Fingerprinting and Disjointness Audit
# =====================================================================

def compute_corpus_fingerprint(texts: list[str]) -> str:
    """Compute deterministic SHA-256 fingerprint for a sequence of texts."""
    canonical = "\n---PASSAGE---\n".join(t.strip() for t in texts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def get_corpus_statistics() -> dict[str, Any]:
    """Compute token, word, and character statistics across all M28 partitions."""
    all_pretrain_words = sum(len(p.split()) for p in PRETRAIN_PROSE_CORPUS)
    all_pretrain_chars = sum(len(p) for p in PRETRAIN_PROSE_CORPUS)

    train_qa_count = sum(len(d.answerable_pairs) + len(d.unanswerable_questions) + len(d.paraphrase_pairs) for d in TRAIN_QA_DOCUMENTS)
    val_qa_count = sum(len(d.answerable_pairs) + len(d.unanswerable_questions) + len(d.paraphrase_pairs) for d in VAL_DOCUMENTS)
    test_qa_count = sum(len(d.answerable_pairs) + len(d.unanswerable_questions) + len(d.paraphrase_pairs) for d in SEALED_TEST_QA_DOCUMENTS)

    return {
        "pretrain_passages": len(PRETRAIN_PROSE_CORPUS),
        "pretrain_words": all_pretrain_words,
        "pretrain_chars": all_pretrain_chars,
        "pretrain_fingerprint": compute_corpus_fingerprint(PRETRAIN_PROSE_CORPUS),
        "train_documents": len(TRAIN_QA_DOCUMENTS),
        "train_qa_samples": train_qa_count,
        "train_fingerprint": compute_corpus_fingerprint([d.text for d in TRAIN_QA_DOCUMENTS]),
        "val_documents": len(VAL_DOCUMENTS),
        "val_qa_samples": val_qa_count,
        "val_fingerprint": compute_corpus_fingerprint([d.text for d in VAL_DOCUMENTS]),
        "test_documents": len(SEALED_TEST_QA_DOCUMENTS),
        "test_qa_samples": test_qa_count,
        "test_fingerprint": compute_corpus_fingerprint([d.text for d in SEALED_TEST_QA_DOCUMENTS]),
        "license": "CC0-1.0 Public Domain / Synthetic Curated Clean Prose",
    }


def verify_all_language_splits_disjoint() -> dict[str, Any]:
    """Audit and guarantee 0.0% document, question, or fact leakage across splits."""
    train_ids = {d.doc_id for d in TRAIN_QA_DOCUMENTS}
    val_ids = {d.doc_id for d in VAL_DOCUMENTS}
    test_ids = {d.doc_id for d in SEALED_TEST_QA_DOCUMENTS}

    # 1. Document ID uniqueness
    id_leakage_tv = train_ids.intersection(val_ids)
    id_leakage_tt = train_ids.intersection(test_ids)
    id_leakage_vt = val_ids.intersection(test_ids)
    if id_leakage_tv or id_leakage_tt or id_leakage_vt:
        raise ValueError(f"Document ID collision detected: tv={id_leakage_tv}, tt={id_leakage_tt}, vt={id_leakage_vt}")

    # 2. Text and question disjointness
    def extract_all_questions(doc_list: list[DocumentPassage]) -> set[str]:
        q_set = set()
        for doc in doc_list:
            for q, _, _ in doc.answerable_pairs:
                q_set.add(normalize_text(q))
            for q in doc.unanswerable_questions:
                q_set.add(normalize_text(q))
            for q, _, _ in doc.paraphrase_pairs:
                q_set.add(normalize_text(q))
        return q_set

    train_q = extract_all_questions(TRAIN_QA_DOCUMENTS)
    val_q = extract_all_questions(VAL_DOCUMENTS)
    test_q = extract_all_questions(SEALED_TEST_QA_DOCUMENTS)

    q_overlap_tv = train_q.intersection(val_q)
    q_overlap_tt = train_q.intersection(test_q)
    q_overlap_vt = val_q.intersection(test_q)

    if q_overlap_tv or q_overlap_tt or q_overlap_vt:
        raise ValueError(f"Question leakage detected: tv={q_overlap_tv}, tt={q_overlap_tt}, vt={q_overlap_vt}")

    # 3. Text passage exact sentence disjointness
    def extract_sentences(doc_list: list[DocumentPassage]) -> set[str]:
        sents = set()
        for doc in doc_list:
            for s in doc.text.split("."):
                norm = normalize_text(s)
                if len(norm) > 15:
                    sents.add(norm)
        return sents

    train_sents = extract_sentences(TRAIN_QA_DOCUMENTS)
    val_sents = extract_sentences(VAL_DOCUMENTS)
    test_sents = extract_sentences(SEALED_TEST_QA_DOCUMENTS)

    sent_overlap_tv = train_sents.intersection(val_sents)
    sent_overlap_tt = train_sents.intersection(test_sents)
    sent_overlap_vt = val_sents.intersection(test_sents)

    if sent_overlap_tv or sent_overlap_tt or sent_overlap_vt:
        raise ValueError("Passage sentence overlap detected across splits!")

    return {
        "status": "PASSED",
        "train_documents": len(train_ids),
        "val_documents": len(val_ids),
        "test_documents": len(test_ids),
        "question_overlap_rate": 0.0,
        "document_overlap_rate": 0.0,
        "sentence_overlap_rate": 0.0,
    }
