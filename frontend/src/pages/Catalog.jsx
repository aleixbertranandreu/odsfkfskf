import { useState, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import ProductCard from '../components/ProductCard';
import products from '../data/products';
import './Catalog.css';

const CATEGORIES = ['mujer', 'hombre', 'kids'];
const SUBCATEGORIES = [...new Set(products.map((p) => p.subcategory))];

export default function Catalog() {
    const [searchParams] = useSearchParams();
    const catParam = searchParams.get('cat') || '';
    const [search, setSearch] = useState('');
    const [sort, setSort] = useState('default');
    const [selectedCats, setSelectedCats] = useState(
        catParam && catParam !== 'novedades' ? [catParam] : []
    );
    const [selectedSubs, setSelectedSubs] = useState([]);
    const [showNew, setShowNew] = useState(catParam === 'novedades');
    const [showFilters, setShowFilters] = useState(true);

    const toggleCat = (cat) => {
        setSelectedCats((prev) =>
            prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
        );
    };

    const toggleSub = (sub) => {
        setSelectedSubs((prev) =>
            prev.includes(sub) ? prev.filter((s) => s !== sub) : [...prev, sub]
        );
    };

    const filtered = useMemo(() => {
        let result = [...products];

        // Search
        if (search.trim()) {
            const q = search.toLowerCase();
            result = result.filter(
                (p) =>
                    p.name.toLowerCase().includes(q) ||
                    p.description.toLowerCase().includes(q) ||
                    p.subcategory.toLowerCase().includes(q)
            );
        }

        // Category filter
        if (selectedCats.length > 0) {
            result = result.filter((p) => selectedCats.includes(p.category));
        }

        // Subcategory filter
        if (selectedSubs.length > 0) {
            result = result.filter((p) => selectedSubs.includes(p.subcategory));
        }

        // New only
        if (showNew) {
            result = result.filter((p) => p.isNew);
        }

        // Sort
        switch (sort) {
            case 'price-asc':
                result.sort((a, b) => a.price - b.price);
                break;
            case 'price-desc':
                result.sort((a, b) => b.price - a.price);
                break;
            case 'name':
                result.sort((a, b) => a.name.localeCompare(b.name));
                break;
            default:
                break;
        }

        return result;
    }, [search, sort, selectedCats, selectedSubs, showNew]);

    const title = catParam === 'novedades'
        ? 'Novedades'
        : catParam
            ? catParam.charAt(0).toUpperCase() + catParam.slice(1)
            : 'Catálogo';

    return (
        <main className="catalog-page">
            <div className="catalog-header">
                <h1>{title}</h1>
                <p>{filtered.length} artículos</p>
            </div>

            {/* Toolbar */}
            <div className="catalog-toolbar">
                <div className="catalog-search">
                    <svg className="catalog-search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" />
                        <path d="m21 21-4.35-4.35" />
                    </svg>
                    <input
                        type="text"
                        placeholder="Buscar productos..."
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        aria-label="Buscar productos"
                    />
                </div>

                <div className="catalog-controls">
                    <button
                        className="filter-toggle"
                        onClick={() => setShowFilters(!showFilters)}
                    >
                        {showFilters ? 'Ocultar filtros' : 'Mostrar filtros'}
                    </button>

                    <div className="catalog-sort">
                        <select
                            value={sort}
                            onChange={(e) => setSort(e.target.value)}
                            aria-label="Ordenar por"
                        >
                            <option value="default">Ordenar por</option>
                            <option value="price-asc">Precio: menor a mayor</option>
                            <option value="price-desc">Precio: mayor a menor</option>
                            <option value="name">Nombre A-Z</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Layout */}
            <div className="catalog-layout">
                {/* Filters */}
                <aside className={`catalog-filters ${!showFilters ? 'hidden' : ''}`} aria-label="Filtros">
                    <div className="filter-group">
                        <h4>Categoría</h4>
                        {CATEGORIES.map((cat) => (
                            <label key={cat} className={`filter-option ${selectedCats.includes(cat) ? 'active' : ''}`}>
                                <input
                                    type="checkbox"
                                    checked={selectedCats.includes(cat)}
                                    onChange={() => toggleCat(cat)}
                                />
                                {cat.charAt(0).toUpperCase() + cat.slice(1)}
                            </label>
                        ))}
                    </div>

                    <div className="filter-group">
                        <h4>Tipo</h4>
                        {SUBCATEGORIES.map((sub) => (
                            <label key={sub} className={`filter-option ${selectedSubs.includes(sub) ? 'active' : ''}`}>
                                <input
                                    type="checkbox"
                                    checked={selectedSubs.includes(sub)}
                                    onChange={() => toggleSub(sub)}
                                />
                                {sub}
                            </label>
                        ))}
                    </div>

                    <div className="filter-group">
                        <h4>Colección</h4>
                        <label className={`filter-option ${showNew ? 'active' : ''}`}>
                            <input
                                type="checkbox"
                                checked={showNew}
                                onChange={() => setShowNew(!showNew)}
                            />
                            Solo novedades
                        </label>
                    </div>
                </aside>

                {/* Grid */}
                <div className="catalog-grid">
                    {filtered.length > 0 ? (
                        filtered.map((p, i) => (
                            <ProductCard key={p.id} product={p} style={{ animationDelay: `${i * 60}ms` }} />
                        ))
                    ) : (
                        <div className="catalog-empty">
                            <h3>No se encontraron resultados</h3>
                            <p>Prueba ajustando los filtros o la búsqueda.</p>
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}
