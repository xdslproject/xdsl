import abc


class AssemblyInstructionArg(abc.ABC):
    """
    Abstract base class for arguments to one line of assembly.
    """

    @abc.abstractmethod
    def assembly_str(self) -> str:
        raise NotImplementedError()
