import { Link } from 'react-router-dom';
import { useCart } from '../context/CartContext';
import './Cart.css';

export default function Cart() {
    const { items, removeItem, updateQuantity, totalItems, totalPrice } = useCart();

    return (
        <main className="cart-page">
            <h1>Carrito</h1>
            <p className="cart-count">{totalItems} {totalItems === 1 ? 'artículo' : 'artículos'}</p>

            {items.length === 0 ? (
                <div className="cart-empty fade-in">
                    <h3>Tu carrito está vacío</h3>
                    <p>Descubre nuestras colecciones y añade tus favoritos.</p>
                    <Link to="/catalogo" className="btn btn-primary">
                        Explorar catálogo
                    </Link>
                </div>
            ) : (
                <div className="fade-in">
                    <div className="cart-items">
                        {items.map((item) => (
                            <div key={`${item.id}-${item.size}`} className="cart-item">
                                <Link to={`/producto/${item.id}`} className="cart-item-image">
                                    <img src={item.image} alt={item.name} />
                                </Link>
                                <div className="cart-item-details">
                                    <div className="cart-item-top">
                                        <div>
                                            <p className="cart-item-name">{item.name}</p>
                                            <p className="cart-item-meta">Talla: {item.size}</p>
                                        </div>
                                        <button
                                            className="cart-item-remove"
                                            onClick={() => removeItem(item.id, item.size)}
                                            aria-label={`Eliminar ${item.name}`}
                                        >
                                            Eliminar
                                        </button>
                                    </div>
                                    <div className="cart-item-bottom">
                                        <div className="cart-qty">
                                            <button
                                                onClick={() => updateQuantity(item.id, item.size, item.quantity - 1)}
                                                aria-label="Reducir cantidad"
                                            >
                                                −
                                            </button>
                                            <span>{item.quantity}</span>
                                            <button
                                                onClick={() => updateQuantity(item.id, item.size, item.quantity + 1)}
                                                aria-label="Aumentar cantidad"
                                            >
                                                +
                                            </button>
                                        </div>
                                        <span className="cart-item-price">{(item.price * item.quantity).toFixed(2)} EUR</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="cart-summary">
                        <div className="cart-summary-row">
                            <span>Subtotal</span>
                            <span>{totalPrice.toFixed(2)} EUR</span>
                        </div>
                        <div className="cart-summary-row">
                            <span>Envío</span>
                            <span>{totalPrice >= 50 ? 'Gratis' : '4.95 EUR'}</span>
                        </div>
                        <div className="cart-summary-row total">
                            <span>Total</span>
                            <span>{(totalPrice >= 50 ? totalPrice : totalPrice + 4.95).toFixed(2)} EUR</span>
                        </div>
                    </div>

                    <div className="cart-actions">
                        <Link to="/checkout" className="btn btn-primary">
                            Finalizar compra
                        </Link>
                        <Link to="/catalogo" className="btn btn-ghost">
                            Seguir comprando
                        </Link>
                    </div>
                </div>
            )}
        </main>
    );
}
