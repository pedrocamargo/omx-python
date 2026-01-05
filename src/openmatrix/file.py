import logging
from typing import Optional, Union, Any

import h5py
import numpy as np
import numpy.typing as npt

from .exceptions import ShapeError, MappingError


class File(h5py.File):
    """
    OMX File class, which contains all the methods for adding, removing, manipulating matrices
    and mappings in an OMX file.
    """

    def __init__(self, name, mode="r", title="", filters=None, **kwargs):
        super().__init__(name, mode, **kwargs)
        self._shape = None
        self.default_filters = filters

    def version(self) -> Optional[str]:
        """
        Return the OMX file format of this OMX file, embedded in the OMX_VERSION file attribute.
        Returns None if the OMX_VERSION attribute is not set.
        """
        if "OMX_VERSION" in self.attrs:
            return self.attrs["OMX_VERSION"]
        else:
            return None

    def create_matrix(
        self,
        name: str,
        shape: Optional[tuple[int, int]] = None,
        title: str = "",
        filters: Union[dict, Any] = None,
        chunks: bool = True,
        obj: Optional[npt.NDArray[Union[np.integer, np.floating]]] = None,
        dtype: Optional[np.dtype] = None,
        attrs: Optional[dict] = None,
    ) -> h5py.Dataset:
        """
        Create an OMX Matrix (Dataset) at the root level. User must pass in either
        an existing numpy matrix, or a shape and a dtype.
        """

        # If object was passed in, make sure its shape is correct
        if self.shape() is not None and obj is not None and obj.shape != self.shape():
            raise ShapeError(f"{name} has shape {obj.shape} but this file requires shape {self.shape()}")

        # Determine dshape and dtype
        dshape = shape
        data = obj
        if obj is not None:
            dshape = obj.shape
            dtype = obj.dtype

        if dshape is None or dtype is None:
            raise ValueError("Shape and dtype must be specified if obj is None")

        # Handle compression
        compression = compression_opts = None
        shuffle = fletcher32 = False

        # If filters is passed (it might be a tables.Filters object or a dict or None)
        # We'll try to parse basic stuff or just use defaults if it's the standard OMX one
        filters = filters or self.default_filters

        if filters:
            # Handle dict
            if isinstance(filters, dict):
                compression = filters.get("complib")
                compression_opts = filters.get("complevel")
                shuffle = filters.get("shuffle")
                fletcher32 = filters.get("fletcher32")

            # Handle object with attributes (like tables.Filters)
            elif hasattr(filters, "complib"):
                compression = filters.complib if filters.complib else compression
                compression_opts = filters.complevel if hasattr(filters, "complevel") else compression_opts
                shuffle = filters.shuffle if hasattr(filters, "shuffle") else shuffle
                fletcher32 = filters.fletcher32 if hasattr(filters, "fletcher32") else fletcher32

        compression = "gzip" if compression == "zlib" else compression

        # Create 'data' group if it doesn't exist
        if not super().__contains__("data"):
            self.create_group("data")

        # create_dataset arguments
        kwargs = {}
        if compression:
            kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
        if shuffle:
            kwargs["shuffle"] = shuffle
        if fletcher32:
            kwargs["fletcher32"] = fletcher32
        if chunks:
            kwargs["chunks"] = chunks

        matrix = super().__getitem__("data").create_dataset(name, shape=dshape, dtype=dtype, data=data, **kwargs)

        if title:
            matrix.attrs["TITLE"] = title

        # Store shape if we don't have one yet
        if self._shape is None:
            storeshape = np.array([matrix.shape[0], matrix.shape[1]], dtype="int32")
            self.attrs["SHAPE"] = storeshape
            self._shape = matrix.shape

        # attributes
        if attrs:
            for key in attrs:
                matrix.attrs[key] = attrs[key]

        return matrix

    def shape(self) -> Optional[tuple[int, int]]:
        """
        Get the one and only shape of all matrices in this File
        """

        # If we already have the shape, just return it
        if self._shape:
            return self._shape

        # If shape is already set in root node attributes, grab it
        if "SHAPE" in self.attrs:
            # Shape is stored as a numpy.array:
            arrayshape = self.attrs["SHAPE"]
            # which must be converted to a tuple:
            self._shape = (arrayshape[0], arrayshape[1])
            return self._shape

        # Inspect the first Dataset object to determine its shape
        data_group = self.data
        if len(data_group) > 0:
            # Get first key
            first_key = list(data_group.keys())[0]
            self._shape = data_group[first_key].shape

            # Store it if we can
            if self.mode != "r":
                storeshape = np.array([self._shape[0], self._shape[1]], dtype="int32")
                self.attrs["SHAPE"] = storeshape
                self.flush()

            return self._shape
        return None

    def list_matrices(self) -> list[str]:
        """List the matrix names in this File"""
        if super().__contains__("data"):
            return list(super().__getitem__("data").keys())
        return []

    def list_all_attributes(self) -> list[str]:
        """Return set of all attributes used for any Matrix in this File"""
        all_tags = set()
        if super().__contains__("data"):
            data_group = super().__getitem__("data")
            for m_name in data_group:
                m = data_group[m_name]
                all_tags.update(m.attrs.keys())
        return sorted(list(all_tags))

    # MAPPINGS -----------------------------------------------
    @property
    def data(self) -> h5py.Group:
        """Return the data group, creating it when writable if missing."""
        if super().__contains__("data"):
            return super().__getitem__("data")
        if self.mode == "r":
            raise MappingError("No matrices available in this file.")
        return self.create_group("data")

    @property
    def lookup(self) -> h5py.Group:
        """Return the lookup group, creating it when writable if missing."""
        if super().__contains__("lookup"):
            return super().__getitem__("lookup")
        if self.mode == "r":
            raise MappingError("No zone mappings available in this file.")
        return self.create_group("lookup")

    def list_mappings(self) -> list[str]:
        """  List all mappings in this file """
        if "lookup" not in self:
            return []
        return list(self.lookup.keys())

    def delete_mapping(self, title) -> None:
        """ Remove a mapping. """
        if "lookup" not in self:
            raise LookupError(f"No such mapping: {title}")

        lookup = self.lookup
        if title not in lookup:
            raise LookupError(f"No such mapping: {title}")

        del lookup[title]
        self.flush()

    def delete_matrix(self, name) -> None:
        """ Remove a matrix."""
        try:
            data_group = super().__getitem__("data")
            del data_group[name]
            self.flush()
        except Exception:
            raise LookupError(f"No such matrix: {name}")

    def mapping(self, title) -> dict[Any, int]:
        """ Return dict containing key:value pairs for specified mapping. """

        if "lookup" not in self:
            raise LookupError(f"No such mapping: {title}")

        lookup = self.lookup
        if title not in lookup:
            raise LookupError(f"No such mapping: {title}")

        entries = lookup[title][:]

        # build reverse key-lookup
        return {k: i for i, k in enumerate(entries)}

    def map_entries(self, title) -> list[Any]:
        """Return a list of entries for the specified mapping."""
        if "lookup" not in self:
            raise LookupError(f"No such mapping: {title}")

        lookup = self.lookup
        if title not in lookup:
            raise LookupError(f"No such mapping: {title}")

        entries = lookup[title][:]
        # Convert to list if it's a numpy array
        if hasattr(entries, "tolist"):
            return entries.tolist()
        return entries

    def create_mapping(self, title, entries, overwrite=False):
        """Create an equivalency index."""

        # Enforce shape-checking
        if self.shape():
            if len(entries) not in self._shape:
                raise ShapeError("Mapping must match one data dimension")

        existing = self.list_mappings()
        if title in existing:
            if overwrite:
                self.delete_mapping(title)
            else:
                raise LookupError(f"{title} mapping already exists.")

        # Ensure lookup group exists when writable and write the mapping
        lookup = self.lookup
        mymap = lookup.create_dataset(title, data=entries)

        return mymap

    # The following functions implement Python list/dictionary lookups. ----
    def __getitem__(self, key):
        """Return a matrix by name, or a list of matrices by attributes"""

        if isinstance(key, str):
            # Direct access to data/lookup or paths
            if key in ["data", "lookup"] or key.startswith("/"):
                return super().__getitem__(key)

            # Check inside 'data' group
            if super().__contains__("data"):
                data_group = super().__getitem__("data")
                if key in data_group:
                    return data_group[key]

            if super().__contains__(key):
                return super().__getitem__(key)

            # If not found
            raise LookupError(f"Key {key} not found")

        if "keys" not in dir(key):
            raise LookupError(f"Key {key} not found")

        # Loop through key/value pairs (attribute lookup)
        mats = []
        if super().__contains__("data"):
            data_group = super().__getitem__("data")
            mats = [data_group[n] for n in data_group]

        for a in key.keys():
            mats = self._getMatricesByAttribute(a, key[a], mats)

        return mats

    def _getMatricesByAttribute(self, key, value, matrices=None):

        answer = []

        if matrices is None:
            if super().__contains__("data"):
                data_group = super().__getitem__("data")
                matrices = [data_group[n] for n in data_group]
            else:
                matrices = []

        for m in matrices:
            if m.attrs is None:
                continue

            # Only test if key is present in matrix attributes
            if key in m.attrs and m.attrs[key] == value:
                answer.append(m)

        return answer

    def __len__(self):
        if super().__contains__("data"):
            return len(super().__getitem__("data"))
        return 0

    def __setitem__(self, key, dataset):
        # We need to determine dtype and shape from the object that's been passed in.
        # This assumes 'dataset' is a numpy object.

        # Check if it's already an h5py dataset (copy?)
        if isinstance(dataset, h5py.Dataset):
            # Copying datasets across files or within file is supported in h5py
            # dest path: /data/key
            if not super().__contains__("data"):
                self.create_group("data")

        # Remove if exists
        if super().__contains__("data"):
            data_group = super().__getitem__("data")
            if key in data_group:
                del data_group[key]

        return self.create_matrix(key, obj=dataset)

    def __delitem__(self, key):
        if super().__contains__("data"):
            data_group = super().__getitem__("data")
            if key in data_group:
                del data_group[key]
                return

        # Try standard delete
        try:
            super().__delitem__(key)
        except Exception as e:
            logging.debug(f"Failed to delete key {key}: {e.args}")

    def __iter__(self):
        """Iterate over the keys in this container"""
        if super().__contains__("data"):
            data_group = super().__getitem__("data")
            for name in data_group:
                yield data_group[name]
        else:
            return iter([])

    def __contains__(self, item):
        # Respect root-level members first (e.g., data/lookup groups or other root-level items)
        if super().__contains__(item):
            return True

        if super().__contains__("data"):
            data_group = super().__getitem__("data")
            return item in data_group

        return False

    # BACKWARD COMPATIBILITY:
    createMapping = create_mapping
    createMatrix = create_matrix
    deleteMapping = delete_mapping
    listMatrices = list_matrices
    listAllAttributes = list_all_attributes
    listMappings = list_mappings
    mapentries = map_entries
    mapEntries = map_entries
