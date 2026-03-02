import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import products from '../data/products';
import './ScanModal.css';

const STATES = { IDLE: 'idle', PREVIEW: 'preview', LOADING: 'loading', RESULT: 'result', ERROR: 'error' };

// Eliminado el mockMatchGarment para usar fetch() real al backend

export default function ScanModal({ isOpen, onClose }) {
    const [state, setState] = useState(STATES.IDLE);
    const [preview, setPreview] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const fileRef = useRef();

    if (!isOpen) return null;

    const handleFile = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            setState(STATES.ERROR);
            setError('Por favor, sube un archivo de imagen válido (JPG, PNG, WebP).');
            return;
        }

        setSelectedFile(file);

        const reader = new FileReader();
        reader.onload = (ev) => {
            setPreview(ev.target.result);
            setState(STATES.PREVIEW);
        };
        reader.readAsDataURL(file);
    };

    const handleAnalyze = async () => {
        if (!selectedFile) {
            setError('No hay imagen seleccionada.');
            setState(STATES.ERROR);
            return;
        }

        setState(STATES.LOADING);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('http://10.20.29.65:8000/api/scan', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('No se pudo analizar la imagen. Verifica que el servidor esté activo.');
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            setResult(data);
            setState(STATES.RESULT);
        } catch (err) {
            setError(err.message || 'Error de conexión');
            setState(STATES.ERROR);
        }
    };

    const handleReset = () => {
        setState(STATES.IDLE);
        setPreview(null);
        setSelectedFile(null);
        setResult(null);
        setError('');
        if (fileRef.current) fileRef.current.value = '';
    };

    const handleClose = () => {
        handleReset();
        onClose();
    };

    return (
        <div className="scan-overlay" onClick={handleClose} role="dialog" aria-modal="true" aria-label="Escanear prenda">
            <div className="scan-modal" onClick={(e) => e.stopPropagation()}>
                <button className="scan-close" onClick={handleClose} aria-label="Cerrar">
                    ✕
                </button>

                <div className="scan-content">
                    <h2 className="scan-title">Escanear Prenda</h2>
                    <p className="scan-subtitle">
                        Sube una foto de la prenda que buscas y encontraremos la más similar en nuestro catálogo.
                    </p>

                    {/* IDLE — Upload */}
                    {state === STATES.IDLE && (
                        <label className="scan-upload-area" htmlFor="scan-file-input" style={{ display: 'block', cursor: 'pointer' }}>
                            <input
                                ref={fileRef}
                                type="file"
                                accept="image/*"
                                onChange={handleFile}
                                id="scan-file-input"
                                aria-label="Seleccionar imagen"
                                style={{ display: 'none' }}
                            />
                            <div className="scan-upload-icon">📷</div>
                            <p className="scan-upload-text">Haz clic o arrastra una imagen aquí</p>
                            <p className="scan-upload-hint">JPG, PNG o WebP — máx. 10MB</p>
                        </label>
                    )}

                    {/* PREVIEW */}
                    {state === STATES.PREVIEW && (
                        <div className="scan-preview">
                            <img src={preview} alt="Vista previa de la prenda" />
                            <div className="scan-actions">
                                <button className="btn btn-secondary" onClick={handleReset}>
                                    Cambiar imagen
                                </button>
                                <button className="btn btn-primary" onClick={handleAnalyze}>
                                    Analizar prenda
                                </button>
                            </div>
                        </div>
                    )}

                    {/* LOADING */}
                    {state === STATES.LOADING && (
                        <div className="scan-loading">
                            <div className="scan-spinner" />
                            <p>Analizando tu prenda...</p>
                            <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '8px' }}>
                                Buscando coincidencias en nuestro catálogo
                            </p>
                        </div>
                    )}

                    {/* RESULT */}
                    {state === STATES.RESULT && result && (
                        <div className="scan-result fade-in">
                            <div className="scan-match-header">
                                <h3>¡Prenda encontrada!</h3>
                                <span className="scan-confidence">{result.confidence}% coincidencia</span>
                            </div>

                            <div className="scan-match-card" style={{ cursor: 'default' }}>
                                <img src={result.image} alt={result.name} />
                                <div className="scan-match-info">
                                    <span className="ref">CÓDIGO: {result.matchId}</span>
                                    <h4>{result.name}</h4>
                                </div>
                            </div>

                            {result.alternatives.length > 0 && (
                                <>
                                    <p className="scan-alts-title">También te puede interesar</p>
                                    <div className="scan-alts-grid">
                                        {result.alternatives.map((alt) => (
                                            <div key={alt.id} className="scan-alt-card" style={{ cursor: 'default' }}>
                                                <img src={alt.image} alt={alt.name} loading="lazy" />
                                                <p>{alt.name}</p>
                                                <span className="alt-price" style={{ fontSize: '0.8rem', color: '#666' }}>CÓDIGO: {alt.id}</span>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}

                            <div style={{ textAlign: 'center' }}>
                                <button className="btn btn-secondary" onClick={handleReset}>
                                    Escanear otra prenda
                                </button>
                            </div>
                        </div>
                    )}

                    {/* ERROR */}
                    {state === STATES.ERROR && (
                        <div className="scan-error fade-in">
                            <div className="scan-error-icon">⚠️</div>
                            <h3>No se pudo analizar</h3>
                            <p>{error}</p>
                            <button className="btn btn-primary" onClick={handleReset}>
                                Intentar de nuevo
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
