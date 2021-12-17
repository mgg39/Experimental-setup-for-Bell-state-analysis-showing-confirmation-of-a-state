import sequence 
import math

import numpy
import random
from numpy import random, outer, add, zeros, multiply
from numpy.random import random_sample

from random import randrange
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sequence.kernel.timeline import Timeline

from sequence.kernel.entity import Entity

from sequence.kernel.timeline import Timeline
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.components.light_source import polarization
from sequence.components.optical_channel import QuantumChannel
from sequence.components.detector import Detector
from sequence.topology.node import Node

from sequence.kernel.entity import Entity
from sequence.utils.encoding import *
from sequence.components.photon import Photon

from ipywidgets import interact
from matplotlib import pyplot as plt

from typing import TYPE_CHECKING
from numpy.random import random_sample

if TYPE_CHECKING:
    from sequence.kernel.timeline import Timeline

from sequence.utils.encoding import polarization

#Mirror module ----------------------------------------------------------------------------------------------
from sequence.utils.encoding import polarization
from sequence.utils.quantum_state import QuantumState


class Photon():
    def __init__(self, name, wavelength=0, location=None, encoding_type=polarization,
                 quantum_state=(complex(1), complex(0))):

        self.name = name
        self.wavelength = wavelength
        self.location = location
        self.encoding_type = encoding_type
        if self.encoding_type["name"] == "single_atom":
            self.memory = None
        self.quantum_state = QuantumState()
        self.quantum_state.state = quantum_state
        self.qstate_key = None
        self.is_null = False

    def entangle(self, photon):

        self.quantum_state.entangle(photon.quantum_state)

    def random_noise(self):

        self.quantum_state.random_noise()

    def set_state(self, state):
        self.quantum_state.set_state(state)

    @staticmethod
    def measure(basis, photon):
 

        return photon.quantum_state.measure(basis)

    @staticmethod
    def measure_multiple(basis, photons):

        return QuantumState.measure_multiple(basis, [photons[0].quantum_state, photons[1].quantum_state])


class Mirror(Entity):

    def __init__(self, name: str, timeline: "Timeline", fidelity=0.98, time_resolution=150):

        Entity.__init__(self, name, timeline)  # Splitter is part of the QSDetector, and does not have its own name
        self.fidelity = fidelity
        self.receivers = []
        # for BB84
        self.start_time = 0
        self.frequency = 0
        self.basis_list = []
        self.photon_counter = 0
        self.time_resolution = time_resolution  # measured in ps


    
    def init(self):

        pass

    """def init(self) -> None:
        Implementation of Entity interface (see base class).

        pass"""

    def get(self, dark_get=False) -> None:
        """Method to receive a photon for measurement.

        Args:
            dark_get (bool): Signifies if the call is the result of a false positive dark count event.
                If true, will ignore probability calculations (default false).

        Side Effects:
            May notify upper entities of a detection event.
        """

        self.photon_counter += 1
        now = self.timeline.now()
        time = round(now / self.time_resolution) * self.time_resolution


NUM_TRIALS = 1000
FREQUENCY = 1e3


class LightSource(Entity):

    def __init__(self, name, timeline, frequency=8e7, wavelength=1550, bandwidth=0, mean_photon_num=0.1,
                 encoding_type=polarization, phase_error=0):


        Entity.__init__(self, name, timeline)
        self.frequency = frequency  # measured in Hz
        self.wavelength = wavelength  # measured in nm
        self.linewidth = bandwidth  # st. dev. in photon wavelength (nm)
        self.mean_photon_num = mean_photon_num
        self.encoding_type = encoding_type
        self.phase_error = phase_error
        self.photon_counter = 0
        # for BB84
        # self.basis_lists = []
        # self.basis_list = []
        # self.bit_lists = []
        # self.bit_list = []
        # self.is_on = False
        # self.pulse_id = 0

    def init(self):

        pass

    # for general use
    def emit(self, state_list, dst: str) -> None:

        time = self.timeline.now()
        period = int(round(1e12 / self.frequency))

        for i, state in enumerate(state_list):

            num_photons = 1

            if random.random_sample() < self.phase_error:
                state = multiply([1, -1], state)

            for _ in range(num_photons):
                wavelength = self.linewidth * random.randn() + self.wavelength
                new_photon = Photon(str(i),
                                    wavelength=wavelength,
                                    location=self.owner,
                                    encoding_type=self.encoding_type,
                                    quantum_state=state)
                
                process = Process(self.owner, "send_qubit", [dst, new_photon])
                
                event = Event(time, process)
                self.owner.timeline.schedule(event)
                self.photon_counter += 1
            time += period


class BeamSplitter(Entity):

    def __init__(self, name: str, timeline: "Timeline", fidelity=1):

        Entity.__init__(self, name, timeline)  # Splitter is part of the QSDetector, and does not have its own name
        self.fidelity = fidelity
        self.receivers = []
        # for BB84
        self.start_time = 0
        self.frequency = 0
        self.basis_list = []

    def init(self) -> None:

        pass

    def get(self, photon: "Photon") -> None:

        assert photon.encoding_type["name"] == "polarization"

        if random_sample() < self.fidelity:
            index = int((self.timeline.now() - self.start_time) * self.frequency * 1e-12)

            if 0 > index or index >= len(self.basis_list):
                return

            res = Photon.measure(polarization["bases"][self.basis_list[index]], photon)
            #get 2
            self.receivers[res].get()

    def set_basis_list(self, basis_list: "List[int]", start_time: int, frequency: int) -> None:

        self.basis_list = basis_list
        self.start_time = start_time
        self.frequency = frequency

    def set_receiver(self, index: int, receiver: "Entity") -> None:

        if index > len(self.receivers):
            raise Exception("index is larger than the length of receivers")
        self.receivers.insert(index, receiver)


class Counter():
    def __init__(self):
        self.count = 0

    def trigger(self, detector, info):
        self.count += 1


class EmittingNode(Node):
    def __init__(self, name, timeline):
        super().__init__(name, timeline)
        self.light_source = LightSource(name, timeline, frequency=80000000, mean_photon_num = 1)
        self.light_source.owner = self

class MiddleNode(Node):

    def __init__(self, name, timeline, direction):
        super().__init__(name, timeline)
        self.mirror = Mirror(name, timeline)
        self.mirror.owner = self
        self.direction = direction 
        self.light_source = LightSource(name, timeline, frequency=80000000, mean_photon_num = 1)
        self.light_source.owner = self
        self.photon_counter = 0
    #src = node1
    def receive_qubit(self, src, qubit):
        #print("mirror received something")
        if not qubit.is_null:
            self.mirror.get()

            y = randrange(100)
            if not (self.mirror.fidelity * 100 ) < y:
                process_photon = Process(self.light_source, "emit",[[qubit.quantum_state.state], self.direction])

                time = self.timeline.now()
                period = int(round(1e12 / self.light_source.frequency))
                event = Event(time, process_photon)
                self.owner.timeline.schedule(event)
                self.photon_counter +=1
                time += period
       

class BSMNode(Node):

    def __init__(self, name, timeline, pre_randomized_direction):
        super().__init__(name, timeline)
        self.mirror = Mirror(name, timeline)
        self.mirror.owner = self
        self.direction = pre_randomized_direction
        self.light_source = LightSource(name, timeline, frequency=80000000, mean_photon_num = 1)
        self.light_source.owner = self
        self.photon_counter = 0
    #src = node1
    def receive_qubit(self, src, qubit):
        #print("BSM received something")
        if not qubit.is_null:
            self.mirror.get()

            y = randrange(100)
            if not (self.mirror.fidelity * 100 ) < y:
                #this is just a n,fix
                process_photon = Process(self.light_source, "emit",[[qubit.quantum_state.state], self.direction])

                time = self.timeline.now()
                period = int(round(1e12 / self.light_source.frequency))
                event = Event(time, process_photon)
                self.owner.timeline.schedule(event)
                self.photon_counter +=1
                time += period

    
class ReceiverNode(Node):
    def __init__(self, name, timeline):
        super().__init__(name, timeline)
        self.detector = Detector(name + ".detector", tl, efficiency=1)
        self.detector.owner = self

    def receive_qubit(self, src, qubit):
        #print("detector received something")
        if not qubit.is_null:
            self.detector.get()
            #print("detector receiving detector")

if __name__ == "__main__":
    runtime = 10e12 
    tl = Timeline(runtime)

    # nodes and hardware
    node1A = EmittingNode("node1A", tl)
    node2A = MiddleNode("node2A", tl, "node3")
    node4A = ReceiverNode("node4A", tl)
    node1B = EmittingNode("node1B", tl)
    node2B = MiddleNode("node2B", tl, "node3")
    node4B = ReceiverNode("node4B", tl)
    
    receiver_nodes = (node4A, node4B)
    str_receiver_nodes = ("node4A", "node4B")

    #not truly random, but will do for now
    length_rn = len(receiver_nodes)
    rand_node_index = random.choice(range(length_rn))
    str_randnode = str_receiver_nodes[rand_node_index]
    randnode =  receiver_nodes[rand_node_index]
    node3 = BSMNode ("node3", tl, str_randnode)



    qc1A = QuantumChannel("qc", tl, attenuation=0, distance=1e3)
    qc2A = QuantumChannel("qc", tl, attenuation=0, distance=1e3)
    qc3 = QuantumChannel("qc", tl, attenuation=0, distance=1e3)
    qc1B = QuantumChannel("qc", tl, attenuation=0, distance=1e3)
    qc2B = QuantumChannel("qc", tl, attenuation=0, distance=1e3)
    qc1A.set_ends(node1A, node2A)
    qc2A.set_ends(node2A, node3)
    qc3.set_ends(node3, randnode)
    qc1B.set_ends(node1B, node2B)
    qc2B.set_ends(node2B, node3)


    # counter
    counterA = Counter()
    counterB = Counter()
    node4A.detector.attach(counterA)
    node4B.detector.attach(counterB)


    # schedule events
    time_bin = int(1e12 / FREQUENCY)
        
    #Process

    process1A = Process(node1A.light_source, "emit", [[((1+0j), 0j)],"node2A"])
    process1B = Process(node1B.light_source, "emit", [[((1+0j), 0j)],"node2B"])

    for i in range(NUM_TRIALS):
            event1A = Event(i * time_bin, process1A)
            event1B = Event(i * time_bin, process1B)
            #Are they happening at the same time? - no idea - time bins
            tl.schedule(event1A)
            tl.schedule(event1B)


    tl.init()
    tl.run()

    print("percent measured A: {}%".format(100 * counterA.count / NUM_TRIALS))
    print("percent measured B: {}%".format(100 * counterB.count / NUM_TRIALS))
